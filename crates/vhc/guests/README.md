# Swarm training guests — experiment modules

This directory is a **separate Cargo workspace** (excluded from the root `daemon-node` workspace) of
`wasm32-unknown-unknown` experiment modules. Each is a `cdylib` the `daemon-train` host instantiates
in a wasmtime sandbox and drives through the tensor ABI (`tabi@1`). The guest is untrusted, sandboxed
code: it can only call the frozen `tabi@1` import vocabulary, under a phase-legality table and
fuel / epoch / memory / op-count budgets (swarm-tensor-abi-spec.md §2, §3.5, §8).

Members:

- `tiny-llama` — the reference LLaMA-family decoder (the shipped preset), a one-line
  `experiment!(TinyLlama)` over `daemon_train_sdk::models::TinyLlama`.
- `test-abi-basic` — a tiny ABI-surface exerciser used by the host runtime tests.

## Building

Guests build via the repo `xtask` (which runs `cargo build --release --target
wasm32-unknown-unknown` in this directory). The `wasm32-unknown-unknown` rust-std is provided by the
flake dev shell, so run everything through it:

```
just build-guests                       # or:
nix develop --command cargo run -p xtask -- build-guests
```

Artifacts land in `guests/target/wasm32-unknown-unknown/release/<name>.wasm` (gitignored). The
`daemon-train` host tests locate them via `SWARM_TEST_GUEST_DIR` if set, else this conventional path,
building on demand if absent. Release modules are size-tuned (`opt-level = "s"`, LTO, strip) and stay
well under a few hundred KB.

## Authoring an experiment

An experiment implements `daemon_train_sdk::Experiment` and is wired to the `da_*` exports with the
`experiment!` macro. The SDK's safe wrappers map 1:1 onto `tabi@1`; nothing here computes tensor math
itself — the host does, behind the ABI.

```rust
use daemon_train_sdk::prelude::*;

struct MyExperiment { w: Param, m: Persistent, v: Persistent /* … */ }

impl Experiment for MyExperiment {
    // Pure function of the config (charges NO host import — HOST-15). Cadence + round modes.
    fn manifest(cfg: &Config) -> Manifest { Manifest::new("my-exp", env!("CARGO_PKG_VERSION"), /*H*/ 4) }

    // da_build: register params/persistents from `[experiment.config]` (Build phase only).
    fn build(cfg: &Config) -> Self { /* Param::new(...), Persistent::local(...)/replicated(...) */ }

    // da_step: one micro-batch forward + backward (accumulate); scale by `ctx.loss_scale(batch)`.
    fn step(&mut self, batch: &Batch, ctx: &StepCtx) { /* … loss.mul_s(ctx.loss_scale(batch)).backward(); */ }

    // da_inner_update: the inner optimizer at the accumulation boundary (adamw_step, then zero_grads).
    fn inner_update(&mut self, inner_step: u32) { /* p.adamw_step(&p.grad(), &m, &v, inner_step+1, …) */ }

    // da_make_update: compress local progress into the opaque round payload (native lane).
    fn make_update(&mut self, round: u64) -> UpdateBuilder { /* profile.make_update(&params) */ }

    // da_ingest_updates: decode + aggregate + outer step over the staged committed set (det lane).
    fn ingest(&mut self, round: u64, updates: &UpdatesView) { /* profile.ingest(&params, updates) */ }
}

daemon_train_sdk::experiment!(MyExperiment);
```

### Lanes (the bit-exactness contract)

- **Native lane** (`Tensor`, GPU/CPU vendor-variant numerics): the forward/backward + local payload
  math in `step`/`inner_update`/`make_update`. NOT a cross-peer numerics reference.
- **Det lane** (`DetTensor`, CPU fp32, bit-exact everywhere via `det-core`): the consensus math in
  `ingest`. The cross-peer **agree-path** (every peer reaching the same post-ingest digest) rides the
  det lane only. `Tensor` and `DetTensor` are separate types, so mixing lanes is a compile error.

### Comm profiles

Reuse a first-party profile (`daemon_train_sdk::profiles`) rather than hand-rolling the compression:

- `SparseLoco` — chunked top-k + 2-bit absmax values + error feedback (the consumer-uplink flagship).
- `DiLoCo` — dense/int8 pseudo-gradient + outer (Nesterov) SGD on a **replicated** det momentum.
- `Demo` — per-step DCT top-k coefficients + sign-SGD ingest.

Each is `manifest(&cfg)` + `make_update(params) -> UpdateBuilder` + `ingest(params, &UpdatesView)`.
Compose one by config (see `TinyLlamaCfg { profile, sparse_loco, diloco, demo }`). Persistent
**class** matters: `class = 1` (`replicated`) state (e.g. DiLoCo outer momentum) is digested +
checkpointed and MUST stay identical across peers; `class = 0` (`local`) state (AdamW moments, error
feedback) is peer-private and rebuilt on rejoin.

### Config

`[experiment.config]` is canonical CBOR. Define a `serde` config struct, parse it with
`cfg.parse::<MyCfg>()`, and provide defaults via `Experiment::defaults()`. Choose dimensions so every
param's element count is a multiple of the profile chunking (`sparse_loco.chunk` / `demo.tile²`) — the
guest does no padding.

## Testing without the host

The SDK's `sim` feature swaps the `tabi@1` extern block for an in-crate CPU backend, so an experiment
is unit-testable natively:

```
nix develop --command cargo test -p daemon-train-sdk --features sim
```

`daemon-train`'s host integration tests then run the same model through the real wasm sandbox
(`cargo test -p daemon-train`), including the `WasmBackend` cross-peer determinism suite.
