# The recorded v1 parity oracle (Phase E sunset; decisions D5)

The two **frozen parity pins** — `tiny_llama_on_barrier_round_reproduces_v1_digests_cpu` and
`catch_up_after_straggle_reproduces_v1_digests_cpu` (`tests/v2_parity.rs`) — and the C3 parity
lanes (`tests/c3_parity.rs`) compare v2 runs against **v1-derived digests and update bytes**.
Before the Phase-E v1 sunset those values were computed **live** by the v1 five-phase driver
(`daemon_vhc_session::WasmBackend` / `daemon_vhc_host::Instance`) on every test run. The sunset
removes that driver, so the oracle was frozen FIRST as this content-addressed bundle — the same
discipline as the A0 frozen fixture (`../a0-frozen-v1/`): the pins survive the v1 driver's
deletion as recorded-oracle regressions, never deleted, never weakened (decisions D5; the E3
brief's critical-care clause).

## Bundle contents

Every file is pinned by blake3 in `expected.json`; the tests re-verify each pin on load.

| Path | What |
|---|---|
| `expected.json` | Every pin + the recorded digests + the schedule + the exact config literals (recorded from the live `TinyLlamaCfg` at capture — nothing transcribed by hand) |
| `model-cfg.v1.cbor` | The exact canonical-CBOR experiment config the v1 oracle was built with (`TinyLlamaCfg { n_layers: 1, seq_len: 9, ..default }`) — spliced verbatim into the v2 guest config so both sides keep configuring identically |
| `updates/v2p-round-{0,1}.bin` | The `v2_parity` oracle's per-round sealed update bytes (CPU `EngineConfig::default()`, all-zero tokens, single-peer self-ingest) |
| `c3/init.f32le.bin` | The c3 oracle's matched init θ (flat f32-le; split by `param_numels`) |
| `c3/trained-round-{0,1}.f32le.bin` | Per-round trained θ (post-inner-steps, pre-ingest) |
| `c3/payload-round-{0,1}.bin` | Per-round committed payload bytes (varied-token schedule) |
| `capture/` | The capture crate — the documented, re-runnable capture command (below) |

## Backend independence

One CPU-lane recording serves the cpu / burn-ndarray / wgpu / cuda parity tiers because the
recorded values are backend-independent by construction: the standing det-lane bit-identity
invariant (refactor §12.1) plus the v1 op-backend parity suites (`burn_backend_parity.rs`,
`burn_wgpu_parity.rs` — every v1 tensor op pinned bit-exact across backends) mean the v1
trajectory (θ, update containers, digests) was identical on every backend. The GPU tiers of
`v2_parity` therefore now assert something strictly stronger than before: v2-on-GPU must
reproduce the recorded v1 trajectory, which holds iff det-lane bit-identity holds.

## Capture command (pre-sunset trees only)

The capture crate path-deps on the v1 driver, so it builds only on a tree that still carries it
(the recorded `captured_from.commit`, `1390f0b7`, or any pre-sunset ancestor). To regenerate:

```
# from a pre-sunset checkout root
cargo run -p xtask -- build-guests          # the oracle runs the checkout's tiny_llama.wasm
cd crates/vhc/host/daemon-vhc-host/tests/fixtures/v1-parity-oracle/capture
CAPTURE_COMMIT=$(git rev-parse --short HEAD) cargo run --release
```

The capture reproduces both live oracles byte-for-byte (the `v2_parity` WasmBackend shape and
the `c3_parity` Instance shape) and rewrites the whole bundle including `expected.json`. On a
post-sunset tree the crate intentionally does not build — regeneration is only meaningful from
the pre-sunset driver; the committed bundle is the permanent record.

## Provenance note on the module bytes

The oracle ran the `tiny_llama.wasm` built at the capture commit (`badbf43b…`, matching that
commit's `guests.blake3` pin, byte-identical across checkout paths via the guests workspace's
rustc shim). The module bytes are not vendored here — unlike A0, the oracle's *outputs* are the
regression subject, and each output file is content-addressed directly.
