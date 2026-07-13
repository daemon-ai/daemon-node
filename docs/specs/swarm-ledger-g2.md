# Swarm P1 — lane G2 ledger (burn-wgpu + VRAM autotune)

Lane record for **G2** of the "Swarm P1 + Transport" program, Wave 2 (program ledger:
`swarm-p1-ledger.md`; program plan conventions are the contract; predecessor lane:
`swarm-ledger-g1.md`). G2 slots **burn-wgpu** (Vulkan/RADV) into the frozen `BurnBackend<B:
AutodiffBackend>` generic seam G1 built, reuses the tolerance harness on a real GPU, extends the
cross-backend det-digest tripwire to cpu-vs-wgpu, and replaces the coarse `assess` estimate with a
real VRAM/RAM budget + OOM-probe micro-batch autotune, wiring real GPU `Hardware` probe numbers.

## Base + branch

- **Repo / worktree:** `daemon-node` @ `/home/j/experiments/daemon-worktree/swarm-engine`.
- **Base commit:** `bd2cb5b` (`mirror(merge-1): freeze Wave-1 interfaces`) on `integrations/swarm-p1`.
- **Branch:** `swarm/g2`.
- **Owns (this wave):** `crates/coprocessor/daemon-train/src/**` (incl. the new `autotune.rs`) +
  `crates/coprocessor/daemon-train/tests/*` + the worker bin `backend` module
  (`src/bin/daemon-train-worker/backend.rs`, pre-split for G2 in Wave 0).
- **Never touched:** the main checkout `/home/j/experiments/daemon`, FROZEN files (root
  `Cargo.toml`, `deny.toml`, `flake.nix` — the `.#vulkan` devShell lane already exists from Wave 0),
  other lanes' dirs (`crates/contracts/daemon-api`, `crates/swarm/*` read-only, `guests/` +
  `daemon-train-sdk` are M1's this wave — coordinate via ledger, never edit).

## Scope (program plan "Wave 2 → G2")

1. **wgpu backend arm** — `#[cfg(feature = "wgpu")] BackendKind::Wgpu` + the single `HostState::new`
   arm instantiating `BurnBackend<Autodiff<Wgpu>>` (the deliberate two-site change G1 flagged);
   device selection via an additive `EngineConfig.gpu_index: Option<u32>`.
2. **Tolerance suite on Vulkan** — reuse `tests/tolerance/mod.rs` with a wgpu backend factory;
   per-op forward/backward parity vs `CpuBackend` within the existing classes; widen a class ONLY
   with a documented per-op justification here (do not blanket-widen).
3. **Cross-backend det-digest tripwire** — extend the equality test to CpuBackend-vs-BurnBackend(wgpu):
   det digests MUST be byte-identical (Risk 2 tripwire); the det lane never touches the GPU.
4. **VRAM autotune + OOM probe** — a real budget from `MetaReport` fields + probed device limits →
   eligibility verdict + chosen micro-batch; OOM path (§10.5): halve micro-batch on an OOM trial,
   floor 1 → ineligible. Map wgpu allocation failure into `TrapCode::BudgetMemory` /
   `ErrorClass::OutOfMemory`. Real GPU `Hardware` numbers in the worker `backend` module.
5. **Tests** — HOST-3 (`absmax_pack_golden`, `absmax_layout_bytes_golden` GPU-vs-CPU + §6.6 layout),
   HOST-8 (`meta_mode_vram_ram_estimates` vs real device), HOST-10 (`ingest_budget_scales_with_count`
   on wgpu), autotune units (`oom_probe_halves_microbatch`, `assess_rejects_insufficient_vram`),
   the extended cross-backend digest, and a tiny-llama round-loop smoke on wgpu.
6. **CI story** — GPU tests detect adapter availability and skip cleanly (loud stderr) when absent,
   so the default gate stays green on GPU-less runners while `.#vulkan` runs the full suite
   (TDD §8.1 tier-2 pattern).

## Determinism story (unchanged from G1 — spec §7.2, program ledger)

The native lane on wgpu is a **tolerance class** (GPU kernels ≠ CpuBackend's fixed-order fp32 tape).
The det lane stays **det-core CPU fp32, bit-exact**: every `det_*` op + compression native
materializes host-side (`to_data`) and runs the identical `det_core` kernel, so the consensus digest
(`digest_state` over post-ingest masters, all written by det ops) is **backend-independent** and
byte-identical across CpuBackend / BurnBackend(ndarray) / BurnBackend(wgpu). The cross-backend
digest test is the early tripwire for a det-lane residency mistake on GPU (Risks 1–2).

## Exported seams (freeze at Merge 2)

### 1. `BackendKind::Wgpu` + device-selection config surface

```rust
// crates/coprocessor/daemon-train/src/runtime.rs
pub enum BackendKind {
    #[default] Cpu,
    #[cfg(feature = "burn-ndarray")] BurnNdarray,
    #[cfg(feature = "wgpu")] Wgpu,           // burn-wgpu autodiff (Vulkan/RADV)
}
pub struct EngineConfig { /* … */ pub gpu_index: Option<u32> }  // additive; Default = None
```

`gpu_index = None` selects `WgpuDevice::DefaultDevice` (best available); `Some(i)` selects
`WgpuDevice::DiscreteGpu(i)`. `BurnWgpuBackend = BurnBackend<Autodiff<Wgpu>>` (feature `wgpu`).
`burn_backend.rs` is now gated `any(feature = "burn-ndarray", feature = "wgpu")` (the generic impl
needs only burn-tensor, which is always on; the two aliases are backend-feature-gated).

### 2. The autotune / assess verdict shape (`autotune.rs` — M1's 160M + B3's worker consume it)

```rust
// crates/coprocessor/daemon-train/src/autotune.rs
pub struct DeviceLimits {                 // what the worker probed / policy caps
    pub vram_mb: u64,
    pub ram_mb: u64,
    pub max_alloc_mb: u64,                 // largest single allocation (wgpu-queryable); 0 = unknown
}
pub struct Autotune {                      // the resource model, built from a MetaReport
    pub fixed_vram_bytes: u64,             // params(storage) + fp32 master + fp32 grad + persistents
    pub act_bytes_per_mb: u64,             // per-micro-batch activation (meta ran at batch=1)
    pub host_ram_bytes: u64,
    pub payload_bytes: u64,
    pub max_tensor_bytes: u64,             // largest single param master (max-alloc check)
}
pub struct AutotuneVerdict {
    pub eligible: bool,
    pub micro_batch: u32,                  // chosen (largest pow2 ≤ max that fits); 0 if ineligible
    pub vram_mb_estimate: u64,             // fixed + micro_batch·act (chosen mb)
    pub ram_mb_estimate: u64,
    pub payload_bytes_estimate: u64,
    pub oom_retries: u32,                  // halvings the OOM probe applied
    pub reasons: Vec<String>,
}
impl Autotune {
    pub fn from_meta(report: &MetaReport) -> Self;
    pub fn verdict(&self, limits: &DeviceLimits, max_microbatch: u32) -> AutotuneVerdict;
}
// OOM probe (§10.5): halve from `start` until the trial succeeds or drops below `floor`.
pub enum ProbeStep { Fits, Oom }
pub fn probe_microbatch(start: u32, floor: u32, trial: impl FnMut(u32) -> ProbeStep)
    -> Option<u32>;                        // None ⇒ even `floor` OOMs → ineligible
```

The frozen `daemon_swarm_run::backend::Assessment` (returned by `WasmBackend::assess`) and
`daemon_swarm_run::protocol::Eligibility` (returned by the worker `assess`) are unchanged shapes; G2
computes the richer `AutotuneVerdict` and maps it into them — the chosen micro-batch rides in the
`reasons` string (`assess`) and the `headroom` map (`micro_batch`, `vram_mb`, `ram_mb`,
`payload_bytes`) on `Eligibility`. `Assessment` has no micro-batch field (frozen at the MVP), so
`AutotuneVerdict` is the authoritative verdict type G2 exports for M1/B3.

### 3. The GPU-skip test convention (`wgpu` feature)

```rust
#[cfg(feature = "wgpu")]
pub fn wgpu_adapter_available() -> bool;   // catch_unwind around a default-device wgpu probe
```

GPU-needing tests early-return with a loud `eprintln!("SKIP … no wgpu adapter")` when this is
`false`, so the default CI gate is green on GPU-less runners while `.#vulkan` runs the full suite
(TDD §8.1: bit-exactness is a CPU property; the GPU only ever affects the native lane).

## Planned slices (base `bd2cb5b`)

1. `mirror(G2): ledger` — this file (commit first).
2. `feat(train): burn-wgpu backend arm behind the OpBackend seam (green)` — `BackendKind::Wgpu`,
   `HostState::new` arm, `EngineConfig.gpu_index`, `BurnWgpuBackend` alias, cfg-gate widening.
3. `feat(train): VRAM autotune + OOM-probe micro-batch sizing (green)` — `autotune.rs` + units
   (`oom_probe_halves_microbatch`, `assess_rejects_insufficient_vram`); `WasmBackend::assess`
   consumes the real budget; real GPU `Hardware` probe + autotune in the worker `backend` module.
3. `feat(train): wgpu tolerance parity + HOST-3 absmax golden on Vulkan (green)`.
4. `feat(train): cross-backend det-digest + HOST-8/10 + tiny-llama smoke on wgpu (green)`.

## Code grounding (burn-wgpu 0.21 / cubecl-wgpu 0.10 API anchors)

- `burn::backend::Wgpu` re-export: `burn-0.21.0/src/backend.rs:34` (`pub use burn_wgpu::Wgpu`,
  feature `wgpu`); `Wgpu<F,I,B> = Fusion<CubeBackend<WgpuRuntime,…>>` `burn-wgpu-0.21.0/src/lib.rs`.
- `WgpuDevice` variants (`DiscreteGpu(usize)` / `DefaultDevice` = best available, honoring
  `CUBECL_WGPU_DEFAULT_DEVICE`): `cubecl-wgpu-0.10.0/src/device.rs:15-51`.
- `WgpuSetup { instance, adapter, device, queue, backend }` from `init_setup::<G>` /
  `init_setup_async`: `cubecl-wgpu-0.10.0/src/runtime.rs:210-258`. `adapter.limits()` /
  `adapter.get_info()` are the wgpu-queryable probe surface.
- `det_core::absmax_pack` §6.6 layout — per chunk `stride = 2 (f16 absmax LE) + ceil(chunk·bits/8)`
  code bytes, codes LSB-first: `det-core/src/lib.rs:424-459` (grounds the HOST-3 layout golden).

## What wgpu actually exposes for VRAM probing (honest inventory)

- **Queryable via `wgpu::Adapter`** (reached through cubecl's `WgpuSetup.adapter`): `get_info()`
  (name / backend / `device_type`) and `limits()` — `max_buffer_size`,
  `max_storage_buffer_binding_size` (the **largest single allocation** — a hard per-tensor ceiling).
- **NOT exposed by wgpu:** total / free VRAM. The Vulkan+WebGPU surface has no device-memory-size
  query, so total VRAM is **not** wgpu-queryable. G2 sources it honestly (documented in
  `autotune.rs` + the worker probe): the OS DRM total when available, else the GPU-governor policy
  cap (§10.5 `vram_cap_mb`), else a conservative estimate flagged in `reasons`. `Hardware.vram_mb`
  carries that value; `max_alloc_mb` (from `limits()`) is the one number that is truly device-honest.
- **GPU count:** cubecl's `init_setup` yields one adapter (the selected physical device). Counting
  all adapters needs `wgpu::Instance::enumerate_adapters(Backends)`, which requires naming raw
  `wgpu` types — a direct `wgpu` dep = a FROZEN root `Cargo.toml` change (not a lane action). So G2
  reports `gpus = 1` when a usable Vulkan adapter initializes (honest: "≥1 usable adapter"), else 0.

<!-- Filled in as slices land. -->

## Landed slices

_(pending)_

## Evidence

_(pending)_

## Deviations / notes for Merge 2 + M2/B3

_(pending)_
