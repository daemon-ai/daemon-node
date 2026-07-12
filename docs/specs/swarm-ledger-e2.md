# Swarm-training MVP — lane E2 ledger (engine / tensor-ABI / guests, Wave 2)

Wave-2 coordination record for lane **E** (`swarm/e2`). Companion to the program ledger
(`swarm-mvp-ledger.md`) and the Wave-1 lane record (`swarm-ledger-e1.md`); this file is lane E's
Wave-2 base/scope/seams/slices record. Read the program ledger's FROZEN-file + file-ownership rules
and the "Merge 1 — frozen interfaces" section first — they bind this lane unchanged. Wave-2 extends
the Merge-1 surfaces **additively only**: the 50-import `tabi@1` subset, the phase-legality table,
the `OpBackend` trait, and the `det-core` signatures grow; nothing existing changes signature or
semantics.

## Base + branch

- **Branch:** `swarm/e2`, forked at `c1432fa` (`mirror(merge-1): freeze cross-lane interfaces`) on
  `integrations/swarm` — Merge 1, all Wave-1 lanes (P1/R1/E1) integrated.
- **Merge target:** `integrations/swarm` (Merge 2). Disjoint file set → conflict-free with the other
  Wave-2 lanes.

## Scope (this lane owns; edits confined here)

| Path | Role |
|---|---|
| `crates/contracts/det-core/` | fixed-order fp32 reference kernels — det lane (ABI §5.9) **+ the shared compression/DCT reference** (§5.8) both sim and host call |
| `crates/contracts/daemon-train-sdk/` | guest SDK: `tabi@1` bindings + wrappers + `Experiment` + `experiment!` + `sim` + **the `profiles` module** (ABI §10.3) |
| `crates/coprocessor/daemon-train/` | host worker runtime (wasmtime host: dispatch, arena, traps, phases, budgets, `OpBackend`/`CpuBackend`, **meta mode**) |
| `guests/` | guest experiment modules (`tiny-llama` — now a real llama-style model — + `test-abi-basic`) + the guests mini-workspace |
| `xtask build-guests` | the guests build subcommand |

FROZEN (never touched): root `Cargo.toml`, `deny.toml`, `flake.nix`, and every other lane's
directories. No new third-party dependency is introduced — every crate used
(`wasmtime`/`burn`/`blake3`/`xxhash-rust`/`thiserror`/`serde`/`ciborium`/`det-core`/`daemon-train-sdk`)
is already pinned in the frozen root `[workspace.dependencies]`.

## Exported seams — FROZEN at Merge 2

### 1. The completed `tabi@1` op vocabulary (v1 frozen vocabulary)

The Merge-1 subset (50 imports, unchanged) plus the Wave-2 additions below. The host `Linker`
(`daemon-train/src/runtime.rs`), the SDK extern block (`daemon-train-sdk/src/abi.rs`), and the
phase-legality table (`daemon-train/src/phase.rs`) agree name-for-name; the additive frozen-surface
sync test (`daemon-train/tests/abi_surface.rs`, extended from Wave 1's phase-table coverage) pins
that agreement.

**Wave-2 additions (all `@1`, additive):**

- **Creation / shape:** `arange`, `reshape`, `transpose`, `slice`, `concat`, `cast`, `gather`,
  `shape_rank`, `shape_dim`, `numel`, `dtype_of`.
- **Elementwise:** `div`, `pow`, `maximum`, `minimum`; scalar-RHS `add_s`, `sub_s`, `div_s`,
  `pow_s`; unary `neg`, `abs`, `sign`, `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `erf`,
  `silu`, `gelu`; `clamp`.
- **Reductions:** `sum_all`, `mean_all`, `max_all`, `min_all`, `sum_dim`, `mean_dim`, `max_dim`,
  `l2_norm`.
- **NN fused:** `embedding`, `rmsnorm`, `layernorm`, `rope`, `flash_attn`, `softmax`.
- **Optimizer steps:** `nadamw_step`, `sgdm_step`, `signum_step`.
- **Compression (native lane, `da_inner_update` + `da_make_update`):** `topk_chunk`,
  `chunk_scatter`, `absmax_pack`, `absmax_unpack`, `dct2`, `idct2`.
- **Det lane (`da_ingest_updates`):** `det_idct2`.
- **Autodiff:** `detach`.

Numeric contract for the compression/DCT natives lives in `det-core` (see seam 3), so the sim and
the host `CpuBackend` share one reference implementation (HOST-1/2/3 goldens).

### 2. Profile config CBOR schemas (ABI §10.3, `daemon_train_sdk::profiles`)

All three profiles are `CommProfile`-shaped library code (`make_update(params) -> UpdateBuilder`;
`ingest(params, &UpdatesView)`; `manifest(&cfg) -> Manifest`). Their config structs are serde types
carried inside the experiment's `[experiment.config]` (canonical CBOR).

```
SparseLocoCfg { h: u32, ef_decay: f64 (β, 0.95), chunk: u32 (4096), topk: u32 (64),
                bits: u32 (2), outer_alpha: f64 (1.0), clip: bool (median-norm) }
DiLoCoCfg     { h: u32, outer_lr: f64 (0.7), momentum: f64 (0.9), nesterov: bool (true),
                quant_bits: u32 (0 = dense fp32, else 8) }
DemoCfg       { momentum_decay: f64 (0.999), chunk: u32 (64 → tile 8), topk: u32 (8),
                sign_lr: f64, wd: f64 (0.1), alpha: f64 (0.2 partial subtraction) }
```

Payload section layout per profile is documented in the profile module docs; the swarm never parses
sections (ABI §4.3).

### 3. `det-core` reference kernel signatures (additive; existing signatures unchanged)

New pure fixed-order fp32 reference kernels (zero-dep, wasm32-clean):

```rust
pub fn dct2(x: &[f32], tile: usize) -> Result<Vec<f32>, DetError>;   // orthonormal 2-D DCT-II per tile²
pub fn idct2(x: &[f32], tile: usize) -> Result<Vec<f32>, DetError>;  // inverse (DCT-III), reconstructs
pub fn topk_chunk(x: &[f32], chunk: usize, k: usize)                 // per-chunk top-k by |magnitude|
    -> Result<(Vec<f32>, Vec<u32>), DetError>;                       // (values[n_chunks*k], idx within chunk)
pub fn absmax_pack(x: &[f32], chunk: usize, bits: u32)               // inverse of det_absmax_unpack
    -> Result<Vec<u8>, DetError>;                                    // §6.6 layout, round-to-nearest code
```

`absmax_pack`/`det_absmax_unpack` round-trip within the codebook quantization error; `dct2`/`idct2`
reconstruct to ≤1e-5 relative; `topk_chunk` returns the `k` largest-magnitude entries per chunk in
descending-magnitude then ascending-index order (ties broken by lower index).

### 4. tiny-llama experiment config schema (guests/tiny-llama)

```
TinyLlamaCfg {
  d_model: u32, n_layers: u32, n_heads: u32, n_kv_heads: u32, head_dim: u32,
  vocab: u32, seq_len: u32, ffn_mult: u32, rope_theta: f64, rmsnorm_eps: f64,
  inner: { lr, beta1, beta2, eps, wd },                 # AdamW inner
  profile: "sparse_loco" | "diloco" | "demo",           # selects the comm profile
  comm: <profile config, tagged by `profile`>,
}
```

Tied embeddings (logits reuse `tok.weight`). `da_manifest` reports `steps_per_round` from the
selected profile (`comm.h` for sparse_loco/diloco, 1 for demo).

## Planned slices (commit order; each lane-scoped green)

1. `mirror(E2): ledger` — this file.
2. `feat(det-core): compression + DCT/topk reference kernels + goldens (green)`.
3. `feat(train-sdk): tabi@1 vocabulary completion — NN/elementwise/shape/reduction/compression
   bindings + sim (green)` and the paired `feat(train): host dispatch + phase table + OpBackend
   extension + name-sync test (green)`.
4. `feat(train-sdk): profiles (sparse_loco/diloco/demo) as libraries + SDK-1..5 tests (green)`.
5. `feat(train): real tiny-llama guest + experiment config + sim loss-decrease test (green)`.
6. `feat(train): meta mode + MetaReport + HOST-8/9 supports (green)`.

## Merge-2 watch list (things integration must verify)

- **The v1 `tabi@1` vocabulary is now frozen** at the union of the Merge-1 subset + the Wave-2
  additions (seam 1). Later waves extend additively only; a fixture, once published, never changes
  for an `op@version`.
- **`burn` is still on the default gate.** Lane E did not lane-split it this wave (the `OpBackend`
  seam still stands; the CPU fake carries the new ops). Track as remaining Wave-2/3 lane-E work.
- **Native-lane autodiff is a semantics reference, not a numerics reference.** The sim + `CpuBackend`
  implement a small reverse-mode tape for the NN ops so tiny-llama trains natively; cross-peer
  bit-identity is a **det-lane** property only (ABI §7) — the profiles' agree-path uses only det ops.
- **Compression natives live in `det-core`.** They are native-lane ops (vendor-variant in the real
  product) but the reference implementation is shared so sim ≡ CPU-fake byte-for-byte; a GPU lane may
  legitimately differ within tolerance (HOST-3 tolerance class).
- **Additive-only.** The Merge-1 frozen surfaces are unchanged; grep the sync test
  (`abi_surface.rs`) for the authoritative name list.
