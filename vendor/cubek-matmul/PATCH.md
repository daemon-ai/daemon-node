# Vendored `cubek-matmul` 0.2.0 — selector shared-memory clamp

This is an unmodified copy of `cubek-matmul` 0.2.0 from crates.io **except** for one change in
the two blueprint selectors (`src/routines/selector/plane.rs`, `src/routines/selector/unit.rs`),
pulled into the host workspace via `[patch.crates-io]` in the root `Cargo.toml`. It exists to
close a select/launch disagreement that is fatal under this workspace's determinism posture:
matmul stage selection never consults the adapter's shared-memory budget, and the launch-side
validator then refuses the kernel (`ResourceLimitError::SharedMemory` in cubecl-wgpu's
`validate_shared`) — on an Apple M4 the unit selector picks a 40,960-byte stage set against the
32,768-byte device limit, mid-round, thirty slices into a training round.

Upstream: tracel-ai/burn#4530 (this exact 40,960 vs 32,768 failure on Apple, open) and #4851
(the same class for FFT, open); unfixed as of 0.2.0 / burn 0.21.0. The **only** upstream path
that consults `client.properties().hardware.max_shared_memory_size` at setup time is the gemv
plane-parallel setup (`src/components/batch/gemv_plane_parallel/setup.rs`), and it refuses
rather than clamps. Burn consumes matmul with autotune off here (determinism posture:
`default-features = false`), so `Strategy::Auto` runs one fixed config with no candidate
fallback — the refusal is a `ComputeFault`, not a reroute.

## The change

Both inferred-selection paths clamp the chosen stage shape to the adapter limit before building
the tiling scheme, walking the dominant contributor down by halves until the lhs + rhs stage
bytes fit:

- `unit.rs`: `clamp_stage_to_shared_memory(...)` (unit-tested in-file with the recorded Apple
  shape: 40,960 B → one halving of the m stage → 24,576 B), applied in `selection()`, with the
  limit threaded from `infer_blueprint_unit`'s existing `ComputeClient`. This is the path the
  wgpu WGSL lanes actually take: without CMMA registration `Strategy::Auto` falls back to
  `SimpleUnit`.
- `plane.rs`: the same walk inline in `infer_blueprint_plane` over
  (`stage_size_m`, `partition_shape_n`, `partition_shape_k`) — the CMMA-capable lanes
  (Vulkan/SPIR-V, Metal/MSL) select through here.

The clamp is **deterministic per adapter**: it is a pure function of the device-property limit
and the otherwise-chosen shape, so identical adapters still select identical configs (the config
is part of what a lane's revision digest already captures). If even a single tile pair exceeds
the limit, the walk floors and the launch validator keeps the last, typed word.

## Upstream-ability

A drop-in PR candidate against tracel-ai/cubek referencing burn#4530/#4851: it extends the
existing gemv precedent (consult `hardware.max_shared_memory_size` at setup) from refusal to
clamped selection, no API change beyond an internal parameter thread. The honesty caveat to
carry upstream: the byte accounting covers the lhs + rhs input stages (the dominant and the
observed-failing terms), not writer/accumulator scratch, so a config sitting exactly at the
limit boundary can still be refused at launch — the validator remains authoritative.
