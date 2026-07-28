# Vendored `cubek-matmul` 0.2.0 — selector shared-memory clamp

This is an unmodified copy of `cubek-matmul` 0.2.0 from crates.io **except** for one change in
the two blueprint selectors (`src/routines/selector/plane.rs`, `src/routines/selector/unit.rs`)
plus the stage-count thread its accounting needs (`src/routines/{double_buffering,
ordered_double_buffering,specialized}.rs`), pulled into the host workspace via
`[patch.crates-io]` in the root `Cargo.toml`. It exists to close a select/launch disagreement
that is fatal under this workspace's determinism posture: matmul stage selection never consults
the adapter's shared-memory budget, and the launch-side validator then refuses the compiled
kernel (`ResourceLimitError::SharedMemory` in cubecl-wgpu's `validate_shared`) — a 40,960-byte
declaration against a 32,768-byte device limit, mid-round, thirty slices into a training round,
on an Apple M4 (Metal) and an RTX 5090 (native DX12) alike.

Upstream: tracel-ai/burn#4530 (this exact 40,960 vs 32,768 failure on Apple, open) and #4851
(the same class for FFT, open); unfixed as of 0.2.0 / burn 0.21.0. The **only** upstream path
that consults `client.properties().hardware.max_shared_memory_size` at setup time is the gemv
plane-parallel setup (`src/components/batch/gemv_plane_parallel/setup.rs`), and it refuses
rather than clamps. Burn consumes matmul with autotune off here (determinism posture:
`default-features = false`), so `Strategy::Auto` runs one fixed config with no candidate
fallback — the refusal is a `ComputeFault`, not a reroute.

## The change

Both inferred-selection paths clamp the chosen stage shape to the adapter limit before building
the tiling scheme, walking the dominant contributor down by halves until the kernel's **full
declared shared-memory budget** fits. The accounting mirrors what `expand_config` + the stage
types actually make the compiled kernel declare — this is the D3-v2 lesson (fit-probe evidence
2026-07-28, RTX 5090/DX12 + M4/Metal, reproduced kernel-by-kernel on the local RADV lane with
instrumented pipeline creation): a first version of this clamp bounded the lhs + rhs input
stages only, and the recorded failing decomposition — tile (4,4,4), partitions (4,2,4), stage
(16,16), f32 — sails under it at 24,576 B while the compiled `matmul_entry` kernel declares
40,960 B, because the writer stage was unaccounted. The budget is now:

- the lhs input stage (`StridedStageMemory` from `lhs_smem_config`), × its global family's
  stage count (`NumStages`: 2 per input on the double-buffered families, (1, 2) on ordered);
- the rhs input stage, likewise;
- the writer stage (`PartitionedStage`, `src/components/global/write/stage.rs`): its
  constructor forces tiles-per-partition to (1, 1), so it declares
  `tile_m · tile_n · stage_m · stage_n` accumulator-stage elements (stage n is pinned to 1 on
  the plane path).

The two seams:

- `unit.rs`: `clamp_stage_to_shared_memory(...)` (unit-tested in-file; the recorded D3-v2
  decomposition is pinned exactly: 40,960 B declared → one halving of the m stage → 24,576 B
  declared, verified end-to-end on a real device with the limit simulated at 32,768 B), applied
  in `selection()`, with the limit threaded from `infer_blueprint_unit`'s existing
  `ComputeClient` and the stage count from the routine's `double_buffering` flag. This is the
  path the wgpu WGSL lanes actually take: without CMMA registration `Strategy::Auto` falls
  back to `SimpleUnit`.
- `plane.rs`: the same walk as `clamp_plane_stage_to_shared_memory(...)` (unit-tested in-file)
  over (`stage_size_m`, `partition_shape_n`, `partition_shape_k`) — the CMMA-capable lanes
  (Vulkan/SPIR-V, Metal/MSL) select through here. The calling routine's stage counts ride in
  through `PlaneTilingBlueprintOptions::num_stages` (`None` = the single-stage families'
  (1, 1); the double-buffered/specialized/ordered routines state theirs at their call sites).

The clamp is **deterministic per adapter**: it is a pure function of the device-property limit
and the otherwise-chosen shape, so identical adapters still select identical configs (the config
is part of what a lane's revision digest already captures), and adapters whose budget already
fits select exactly what they always did. If even a single tile pair exceeds the limit, the
walk floors and the launch validator keeps the last, typed word.

## Upstream-ability

A drop-in PR candidate against tracel-ai/cubek referencing burn#4530/#4851: it extends the
existing gemv precedent (consult `hardware.max_shared_memory_size` at setup) from refusal to
clamped selection, no API change beyond an internal parameter thread and one defaulted public
options field. The honesty caveat to carry upstream: the byte accounting covers the lhs + rhs
input stages (× their stage counts) and the `PartitionedStage` writer — every shared array the
burn-driven `A @ B` kernels declare — but not a fused acc INPUT stage (only C-input matmuls
declare one) nor the per-stage alignment rounding (≤ the swizzle atom), so a config sitting
exactly at the limit boundary can still be refused at launch — the validator remains
authoritative.
