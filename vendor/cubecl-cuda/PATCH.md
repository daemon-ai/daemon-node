# Vendored `cubecl-cuda` 0.10.0 — allocation fence-visibility patch

This is an unmodified copy of `cubecl-cuda` 0.10.0 from crates.io **except** for one change in
`src/compute/server.rs`, pulled into the host workspace via `[patch.crates-io]` in the root
`Cargo.toml`. It exists to close a fence-visibility gap in the `compute@2` deferred-error contract
(ABI-spec §15, decisions D8): allocation failures on the CUDA backend were readback-visible only,
because the alloc path panicked instead of recording the error where a fence could drain it.

## The change

`CudaServer::initialize_memory` reserved device memory with `command.reserve(size).unwrap()`. On a
fallible reservation this panics on the device-stream thread; cubecl catches the per-task panic and
drops it, leaving **no** error state for `RunnerClient::sync`/`fence` to observe. The launch path
(`launch` → `stream.current().errors.push(err)`) and the write path (`command.error(err.into())`)
already record their failures on the stream error queue instead of panicking.

The patch makes the alloc path behave like those two:

```rust
match command.reserve(size) {
    Ok(reserved) => command.bind(reserved, memory),
    Err(err) => command.error(err.into()),   // was: command.reserve(size).unwrap()
}
```

`ServerError: From<IoError>` already exists, so `err.into()` is the same conversion the write path
uses. A recorded error is drained by the next `sync`/`flush` (the fence), which returns
`ServerError::ServerUnhealthy` — surfacing host-side as the typed `ComputeError::Device`
(`TrapCode::ComputeFault`). This covers both reservation-failure classes, since both arrive as
`IoError` at `command.reserve`:

- the **host-side pool-cap rejection** (`IoError::BufferTooBig` when no pool's `max_alloc_size`
  accepts the request, before any `cuMemAlloc`); and
- a **genuine driver `CUDA_ERROR_OUT_OF_MEMORY`**, which the GPU storage layer
  (`storage/gpu.rs::alloc`) also maps to `IoError::BufferTooBig` and returns.

## Upstream-ability

Small and self-contained — a drop-in candidate for an upstream PR against tracel-ai/cubecl: it only
replaces an `.unwrap()` with the crate's own established error-recording idiom, no API or behavior
change beyond turning a dropped panic into a queued, drainable error. The one honesty caveat to
carry upstream: the storage layer collapses a real driver OOM into the same `BufferTooBig` variant
as the pool-cap case, so the drained fault does not by itself distinguish "driver engaged" from
"host-side cap"; a faithful diagnostic would additionally thread the driver error class through
`storage/gpu.rs::alloc`.
