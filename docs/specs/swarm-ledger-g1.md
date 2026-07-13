# Swarm P1 — lane G1 ledger (BurnBackend seam, burn-ndarray first)

Lane record for **G1** of the "Swarm P1 + Transport" program, Wave 1 (program ledger:
`swarm-p1-ledger.md`; program plan conventions are the contract). G1 builds the `BurnBackend`
implementation of the frozen `OpBackend` trait on the **burn-ndarray** CPU backend + autodiff,
plus the per-op **tolerance-class harness** (defines TDD HOST-3's machinery) and the
**cross-backend det-digest equality** tripwire (HOST-7 extension, program Risks 1–2). G2 slots
`burn-wgpu` into the same generic seam in Wave 2.

## Base + branch

- **Repo / worktree:** `daemon-node` @ `/home/j/experiments/daemon-worktree/swarm-engine`.
- **Base commit:** `d71839a` (`mirror(P1-prog): Wave-0 scaffold record`) on `integrations/swarm-p1`.
- **Branch:** `swarm/g1`.
- **Owns (this wave):** `crates/coprocessor/daemon-train/src/{backend.rs, burn_backend.rs,
  wasm_backend.rs, meta.rs, runtime.rs, lib.rs}` and `crates/coprocessor/daemon-train/tests/*`.
  The worker `backend` bin module (`src/bin/daemon-train-worker/backend.rs`) is G2's; `transport.rs`
  is B3's — untouched. `crates/contracts/{det-core,daemon-train-sdk}` only if a genuine fix surfaces
  (none needed this wave).
- **Never touched:** the main checkout `/home/j/experiments/daemon`, FROZEN files (root
  `Cargo.toml`, `deny.toml`, `flake.nix`), and other lanes' directories.

## Scope (program plan "Wave 1 → G1")

1. `BurnBackend<B: burn::tensor::backend::AutodiffBackend>: OpBackend` in new `burn_backend.rs`
   behind `#[cfg(feature = "burn-ndarray")]`. Generic so wgpu (G2) is a type-parameter swap.
2. Map every native-lane op the trait carries onto burn tensor ops; **use burn's own autodiff**
   (`Tensor::backward` / `Tensor::grad`) instead of the CPU tape (HOST-9 for the native lane).
3. Det lane stays **det-core CPU fp32** — device→host materialization at every det op, exactly like
   `CpuBackend` (ABI §5.9 residency contract). Compression natives delegate to det-core host-side.
4. Backend selector so `WasmBackend` (and the round loop) can be driven by either `CpuBackend` or
   `BurnBackend(ndarray)` (needed for the cross-backend digest test).
5. Tolerance-class harness (op → class → rtol/atol) comparing `BurnBackend(ndarray)` vs `CpuBackend`
   forward outputs + backward grads. Backend pair is parametric so G2 reuses it for wgpu.
6. Cross-backend det-digest equality test: run the tiny-llama round loop with one `CpuBackend` peer
   and one `BurnBackend(ndarray)` peer; assert the det-lane digests are equal every round.
7. Named tests: `abi_adamw_step_matches_burn`, `abi_matmul_backward`,
   `grads_invariant_to_accumulation_split`, per-op tolerance tests, the cross-backend digest test;
   all existing daemon-train tests green on default AND `--features burn-ndarray`.

## Determinism story (why this is sound — program ledger "Determinism story", spec §7.2)

- **Native lane = tolerance class.** burn's autodiff (ndarray, and later wgpu) is not bit-wise
  identical to `CpuBackend`'s fixed-order fp32 tape. Forward outputs and backward grads are compared
  under per-op rtol/atol classes, never exact equality (HOST-3 machinery).
- **Det lane = bit-exact everywhere.** The `det_*` ops and the compression natives delegate to the
  same `det_core` kernels host-side on materialized `Vec<f32>` — byte-identical to `CpuBackend`.
- **The consensus digest is backend-independent by construction.** `WasmBackend::digest_of` samples
  the post-ingest canonical state (param fp32 masters + replicated persistents). For all three P1
  profiles the post-ingest masters are written **only** by det-lane ops (DiLoCo-family rebase:
  `det_reset_param_to_base` then `det_axpy_param`; per-step demo: det sign-SGD + decoupled decay at
  ingest). The round-base snapshot (`param_round_base`) is taken at the ingest barrier from the
  post-ingest master, so it too is det-exact and identical across backends inductively. Therefore:
  given the **same committed set** in the same record order, `CpuBackend` and `BurnBackend(ndarray)`
  produce **equal** digests every round — while their losses / payloads differ (native drift). This
  is exactly the cross-backend digest test, the early tripwire for a det-lane residency mistake
  (program Risks 1–2).

## Exported seams (freeze at Merge 1)

### 1. `BurnBackend` construction surface + backend-generic bound

```rust
// crates/coprocessor/daemon-train/src/burn_backend.rs  (#[cfg(feature = "burn-ndarray")])
pub struct BurnBackend<B: burn::tensor::backend::AutodiffBackend> { /* private */ }

impl<B: burn::tensor::backend::AutodiffBackend> BurnBackend<B> {
    pub fn new() -> Self;                    // uses B::Device::default()
    pub fn with_device(device: B::Device) -> Self;
}

impl<B: burn::tensor::backend::AutodiffBackend> OpBackend for BurnBackend<B> { /* ... */ }
```

The single bound is `B: burn::tensor::backend::AutodiffBackend`. G2 instantiates
`BurnBackend<burn::backend::Autodiff<burn::backend::Wgpu>>` with no other change; G1 ships
`BurnBackend<burn::backend::Autodiff<burn::backend::NdArray>>` as the type alias
`BurnNdarrayBackend` (feature-gated).

### 2. Backend selector on `EngineConfig`

```rust
// crates/coprocessor/daemon-train/src/runtime.rs
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendKind {
    #[default] Cpu,                          // the frozen fixed-order fp32 tape (MVP behavior)
    #[cfg(feature = "burn-ndarray")] BurnNdarray,
    // G2 adds `#[cfg(feature = "wgpu")] Wgpu` here — one arm in `HostState::new`.
}

pub struct EngineConfig { /* ... existing ... */ pub backend: BackendKind }
```

`EngineConfig::default().backend == BackendKind::Cpu`, so every existing construction is unchanged
(additive field with a `Default`). `WasmBackend` selects the backend purely through
`WasmBackendConfig.engine.backend`; nothing burn leaks across the `TrainerBackend` seam.

### 3. Tolerance-class harness API (how G2 parametrizes it)

Lives in `tests/tolerance.rs` (a `#[path]`-included module reused by the burn-ndarray test file);
G2's wgpu test includes the same module and swaps the backend factory.

```rust
pub enum OpClass { Exact, Shape, Elementwise, MatmulReduce, Normalization, Attention, Optimizer }
pub struct Tol { pub rtol: f32, pub atol: f32 }
pub fn tol_for(class: OpClass) -> Tol;                 // the per-class table
pub fn assert_close(got: &[f32], want: &[f32], class: OpClass, ctx: &str);
```

The comparison runner drives the same op on a `CpuBackend` and a `BurnBackend(ndarray)` over
pinned-seed fixed inputs (seed `0xDAE07E57`) and asserts forward + backward within class. G2 reuses
the runner with a `BurnBackend(wgpu)` factory.

### Tolerance table (op → class → rtol/atol)

| op(s) | class | rtol | atol |
|---|---|---|---|
| create/zeros/clone/view/write/free, reshape, transpose, slice | `Shape` (exact bit-for-bit) | 0 | 0 |
| add, add_bias, sub, mul, mul_s, relu | `Elementwise` | 1e-5 | 1e-6 |
| matmul, embedding | `MatmulReduce` | 1e-4 | 1e-5 |
| rmsnorm, softmax, silu, rope | `Normalization` | 1e-4 | 1e-5 |
| cross_entropy, flash_attn | `Attention` | 2e-4 | 2e-5 |
| adamw_step | `Optimizer` | 2e-4 | 2e-5 |
| all `det_*` + compression natives (dct2/idct2/topk_chunk/absmax_pack/unpack) | `Exact` | 0 | 0 |

Values are measured from the actual ndarray-vs-cpu deltas on the fixed fixtures (headroom left for
wgpu). `Exact` classes assert byte-identity (both delegate to det-core / are pure data moves).

## Planned slices (TDD-style, each `feat(train): … (green)`)

1. `mirror(G1): ledger` — this file (commit first).
2. `feat(train): BurnBackend forward+autodiff over the OpBackend seam (green)` — `burn_backend.rs`,
   backend selector, unit tests in the module.
3. `feat(train): cross-backend tolerance harness + HOST-9 parity (green)` — `tests/tolerance.rs`,
   `tests/burn_backend_parity.rs` (`abi_matmul_backward`, `abi_adamw_step_matches_burn`,
   `grads_invariant_to_accumulation_split`, per-op tolerance tests).
3. `feat(train): cross-backend det-digest equality tripwire (green)` — extend
   `tests/wasm_backend_determinism.rs` with the CpuBackend-vs-BurnBackend(ndarray) digest test.

## Code grounding (burn 0.21 API anchors, `~/.cargo/registry/.../burn-*-0.21.0`)

- `AutodiffBackend` trait + `Tensor::backward`/`Tensor::grad`: `burn-backend-0.21.0/src/backend/base.rs:213-390`,
  `burn-tensor-0.21.0/src/tensor/api/autodiff.rs:5-24`.
- `Tensor::from_data(data, &device)` / `to_data()` / `into_data()`: `burn-tensor-.../api/base.rs:1898-1955`;
  `TensorData::new` / `to_vec::<f32>()`: `burn-backend-0.21.0/src/data/tensor.rs:59,172`.
- ops: `matmul` `numeric.rs:915`; `add/sub/mul/mul_scalar/neg` `numeric.rs:43,100,249,277,301`;
  `mean_dim` `numeric.rs:390`; `sum` `numeric.rs:361`; `powf_scalar/recip/exp/log/sqrt`
  `float.rs:772,59,919,935,941`; `select` `base.rs:1641`; `narrow` `base.rs:2310`; `swap_dims`
  `base.rs:472`; `reshape` `base.rs:386`; `cat` `base.rs:2182`; `require_grad`/`detach`
  `float.rs:329,320`; activations `relu/softmax/sigmoid/silu` `activation/base.rs:10,171,303,360`.
- backend re-exports `burn::backend::{Autodiff, NdArray}`: `burn-0.21.0/src/backend.rs:11,22`.

## Deviations / notes for Merge 1 + G2

- **Tensor representation:** BurnBackend stores every tensor as a flat rank-1 `Tensor<B, 1>` plus a
  cached host `Vec<f32>` (so `OpBackend::view` stays a cheap `&[f32]`); shape-carrying ops
  (`matmul`, `transpose`, `slice`, `rmsnorm`, `softmax`, `flash_attn`, `add_bias`) reshape to the
  needed const rank internally and flatten back. `reshape@1` is an autodiff identity (tensor clone).
  This trades memory (2× host copy) for a clean `view`; fidelity-over-speed this wave (G2 may cache
  lazily on GPU).
- **`transpose`/`slice` rank coverage:** implemented for ranks 1–4 (the tiny-llama model uses ranks
  2 and 4). Higher ranks trap `ShapeMismatch` via the runtime shape validation before reaching the
  backend; if a future preset needs rank ≥5, extend the match arms.
- **`adamw_step`** is implemented with burn tensor ops (f32) — a genuine native-lane divergence from
  `CpuBackend`'s f64 accumulation, covered by the `Optimizer` tolerance class. The det-lane rebase
  discards this drift at ingest, so it never reaches the consensus digest.
- **GPU-native compression / det kernels** (dct2/idct2/topk_chunk/absmax_pack on-device) are future
  work: BurnBackend delegates them to det-core host-side exactly like CpuBackend (materialize →
  det-core → re-insert). Recorded here for G2/M-lanes.
- **`abi_adamw_step_matches_burn`** interpretation: the ABI's fused `adamw_step` (burn native path)
  is asserted equal to the `CpuBackend` reference implementation within the `Optimizer` tolerance
  class over the pinned fixtures — "burn" being the native engine under test. (A full burn-optim
  `AdamW` oracle is heavier module plumbing; deferred, the reference impl is the closed-form AdamW.)
