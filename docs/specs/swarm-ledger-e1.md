# Swarm-training MVP — lane E1 ledger (engine / tensor-ABI / guests)

Wave-1 coordination record for lane **E** (`swarm/e1`). Companion to the program ledger
(`swarm-mvp-ledger.md`); this file is lane E's own base/scope/seams record. Read the program
ledger's FROZEN-file + file-ownership rules first — they bind this lane unchanged.

## Base + branch

- **Branch:** `swarm/e1`, forked from the Wave-0 scaffold tip.
- **Base commit:** `d442cd8` (`docs(specs): swarm MVP program ledger`) on `integrations/swarm`.
- **Merge target:** `integrations/swarm` (disjoint file set → conflict-free with lanes P and R).

## Scope (this lane owns; edits confined here)

| Path | Role |
|---|---|
| `crates/contracts/det-core/` | fixed-order fp32 deterministic kernels (ABI §5.9 semantics) |
| `crates/contracts/daemon-train-sdk/` | guest experiment SDK (`tabi@1` bindings + wrappers + `Experiment` + `experiment!` + `sim`) |
| `crates/coprocessor/daemon-train/` | host worker runtime (wasmtime side: engine, arena, traps, phases, lifecycle, budgets) |
| `guests/` | guest experiment modules (`tiny-llama`, `test-abi-basic`) + the guests mini-workspace |
| `xtask build-guests` | the guests build subcommand (shared tooling; coordinate any signature change) |

FROZEN (never touched by this lane): root `Cargo.toml`, `deny.toml`, `flake.nix`, and every other
lane's directories. No new third-party dependency is introduced — every crate used here
(`wasmtime`, `burn`, `blake3`, `xxhash-rust`, `thiserror`, `serde`, `ciborium`, `det-core`,
`daemon-train-sdk`) is already pinned in the frozen root `[workspace.dependencies]`.

## Dependency note (parallel lanes)

Lanes P (proto/coordinator) and R (transport/supervisor) run in parallel; this lane uses **none**
of their unfinished APIs. `det-core` is zero-dep; the SDK is serde+ciborium (+`det-core` under
`sim`); `daemon-train`'s host runtime this wave needs **no** proto/transport types (the worker
protocol / CBOR-stdio integration is Wave 3). `daemon-train` keeps the scaffold's
`daemon-swarm-proto` dependency declared but unused (Wave-3 seam), mirroring the Wave-0 machete
ignore.

## Exported seams — FROZEN at Merge 1 (record the exact surface here)

### 1. `det-core` kernel signatures (fixed-order fp32)

```rust
pub fn det_sum(xs: &[&[f32]]) -> Result<Vec<f32>, DetError>;              // elementwise, array order
pub fn det_l2norm(x: &[f32]) -> f32;                                      // fixed-order accumulation
pub fn det_axpy(y: &mut [f32], alpha: f64, x: &[f32]) -> Result<(), DetError>; // y += (alpha as f32)*x
pub fn det_scale(x: &[f32], alpha: f64) -> Vec<f32>;                      // x * (alpha as f32)
pub fn det_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, DetError>;
pub fn det_sub(a: &[f32], b: &[f32]) -> Result<Vec<f32>, DetError>;
pub fn det_mul(a: &[f32], b: &[f32]) -> Result<Vec<f32>, DetError>;
pub fn det_sign(x: &[f32]) -> Vec<f32>;
pub fn det_chunk_scatter_add(acc: &mut [f32], vals: &[f32], idx: &[u32], chunk: usize)
    -> Result<(), DetError>;                                             // in-place, fixed order
pub fn det_chunk_scatter(vals: &[f32], idx: &[u32], chunk: usize, out_len: usize)
    -> Result<Vec<f32>, DetError>;                                       // allocating form
pub fn det_absmax_unpack(packed: &[u8], chunk: usize, bits: u32)
    -> Result<Vec<f32>, DetError>;                                       // 1/2/4/8-bit, fp16 codebook
```

Scalar cast rule (frozen): `f64` scalars are cast to `f32` **inside** the kernel (one cast site
shared by host + sim), per ABI §5.9. `det_absmax_unpack` layout (frozen, ABI §6.6): per chunk a
little-endian `f16` absmax scalar (2 bytes) then `chunk` codes of `bits` width, LSB-first,
chunk-major, zero-padded to a byte boundary; dequant is the symmetric linear codebook
`absmax · (2·code/(2^bits − 1) − 1)`.

### 2. `tabi@1` import list implemented so far (Merge-1 vocabulary subset)

Host-linked imports wired in `daemon-train` this wave (all `@1`):

`param, persistent, det_persistent, drop, param_round_base, backward, grad, zero_grads, assign,
zeros, add, sub, mul, mul_s, matmul, relu, cross_entropy, scalar, metric, log, abi_minor,
adamw_step, batch_tokens, batch_size, batch_seq_len, upd_new, upd_push_bytes, upd_push_tensor,
upd_sections, upd_kind, upd_bytes_len, upd_read_bytes, upd_tensor, det_zeros, det_sum, det_scale,
det_l2norm, det_sign, det_absmax_unpack, det_chunk_scatter_add, det_param, det_reset_param_to_base,
det_axpy_param` — 43 imports. This is a deliberate subset of the full 108-import `tabi@1` (ABI §5):
the compression natives (`topk_chunk`, `absmax_pack`, `dct2/idct2`), the full elementwise/reduction
NN vocabulary, and the remaining fused optimizers land in later waves, additively (§9). The SDK's
extern block and the host Linker agree on this subset name-for-name.

### 3. SDK `Experiment` trait + `experiment!` macro surface

```rust
pub trait Experiment: Sized {
    fn manifest(cfg: &Config) -> Manifest;
    fn build(cfg: &Config) -> Self;
    fn step(&mut self, batch: &Batch, ctx: &StepCtx);
    fn inner_update(&mut self, inner_step: u32);
    fn make_update(&mut self, round: u64) -> UpdateBuilder;
    fn ingest(&mut self, round: u64, updates: &UpdatesView);
}
daemon_train_sdk::experiment!(MyExp);   // → da_abi/da_manifest/da_defaults/da_alloc/da_free
                                        //   + da_build/da_step/da_inner_update/da_make_update/
                                        //     da_ingest_updates trampolines over a guest singleton
```

Core types (frozen surface): `Tensor`, `DetTensor` (separate type ⇒ lane errors are compile
errors), `Param`, `Persistent`, `DetPersistent` (all `Drop` over `drop@1` for step handles),
`Batch`, `StepCtx { inner_step, mb_index, mb_count, step_seqs }` with `loss_scale()`, `Config`
(CBOR view), `Manifest`, `UpdateBuilder`, `UpdatesView`. Feature `sim` swaps the extern block for
an in-crate CPU backend backed by `det-core` (ABI §10.4).

### 4. `daemon-train` `OpBackend` trait (Wave-2 burn seam) + host runtime entry API

```rust
pub trait OpBackend { /* create/shape/math/nn/autodiff/optimizer/det-lane/container ops */ }
pub struct CpuBackend;                 // Wave-1 fake: plain Vec<f32>, det ops via det-core

pub struct Worker { /* wasmtime Engine + InstancePre + host state */ }
impl Worker {
    pub fn new(cfg: &EngineConfig) -> Result<Self, TrainError>;
    pub fn load_module(&self, wasm: &[u8]) -> Result<ModuleHandle, TrainError>;
    pub fn instantiate(&self, module: &ModuleHandle) -> Result<Instance, TrainError>;
    // lifecycle: abi_gate → manifest → build → step/inner_update/make_update/ingest
}
```

The `OpBackend` trait is the **only** seam Wave 2 replaces (slot burn/CubeCL behind it); the
wasmtime host-call dispatch, handle arena, trap taxonomy, phase table, and budgets are lane-E
stable. Wave 2 must NOT change the `tabi@1` import names or the phase-legality table without a
Merge coordination note.

## Planned slices (commit order; each green for `-p det-core -p daemon-train-sdk -p daemon-train`,
plus `--features sim` for the SDK)

1. `mirror(E1): ledger` — this file.
2. `feat(det-core): fixed-order fp32 kernels + golden/property tests (green)`.
3. `feat(train-sdk): tabi@1 bindings, wrapper types, Experiment + experiment! macro (green)`.
4. `feat(train-sdk): sim backend (det-core) + toy 2-layer experiment end-to-end (green)`.
5. `feat(train): engine config + handle arena + trap taxonomy + phase table (green)`.
6. `feat(train): lifecycle driver + OpBackend fake + budgets + T3 re-instantiation (green)`.
7. `feat(train): guest lifecycle integration (tiny-llama + test-abi-basic) (green)`.

## Merge-1 watch list (things integration must verify)

- **`.wasm` test artifacts** land in `guests/target/wasm32-unknown-unknown/release/<name>.wasm`
  (the guests mini-workspace's own target dir, gitignored). `daemon-train`'s guest-integration
  tests locate them via the env override `SWARM_TEST_GUEST_DIR` if set, else the conventional path
  relative to the crate manifest; if the artifact is absent the test **builds it on demand**
  (`cargo build --release --target wasm32-unknown-unknown` inside `guests/`, exactly what
  `xtask build-guests` does), so `cargo test --workspace` never silently skips. First run is slow
  (wasm rebuild); the dev-shell `wasm32-unknown-unknown` rust-std is required (bare host cargo
  fails — program ledger note).
- The `tabi@1` subset (seam 2) and the phase-legality table are the frozen ABI surface; later
  lanes/waves extend them **additively** only.
- `det-core` is on the wasm32 path (via the SDK `sim`→host shared kernels intent); it stays
  zero-dep. Do not add crates to it.
