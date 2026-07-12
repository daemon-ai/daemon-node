# Swarm-training MVP — lane R1 ledger (runtime / transport / supervisor)

Wave-1 coordination record for lane **R** (`swarm/r1`). Companion to the program ledger
[`swarm-mvp-ledger.md`](swarm-mvp-ledger.md); read that first for the branch map, file-ownership
table, and the frozen-file rule. This file records what lane R builds, the seams it exports (frozen
at Merge 1), and the placeholder types that Merge 1 must swap for `daemon-swarm-proto` (lane P).

## Base + branch

- **Branch:** `swarm/r1`, forked from `d442cd8` (`docs(specs): swarm MVP program ledger`) on
  `integrations/swarm`.
- **Merge target:** `integrations/swarm` (disjoint file set → conflict-free by construction).

## Scope (this wave)

Lane R owns `crates/swarm/daemon-swarm-net`, `crates/swarm/daemon-swarm-run`, and
`crates/coprocessor/daemon-train-client` (plus `tests/` + `bins/` swarm surfaces in later waves).
Wave 1 delivers the **seams** the round loop (Wave 2) and the worker (Wave 3) build over — no
round loop, no real engine, no egress.

| Slice | Crate | Spec | TDD |
|---|---|---|---|
| `SwarmTransport` (control + payload planes) | `daemon-swarm-net` | §7.1 | NET-5/6/8 |
| `LoopbackGossip` (in-process control plane) | `daemon-swarm-net` | §7.1 | NET-6 |
| `FsPayloadStore` (fs payload plane + retention + stat) | `daemon-swarm-net` | §7.1, §7.4, §6.4 | NET-8 |
| `ReceiptProducer` (signed `StorageReceipt` evidence) | `daemon-swarm-net` | §6.4 I6, §7.1 | NET-1 (fs half) |
| Artifact fetch (`file://` + blake3, scheme-dispatch) | `daemon-swarm-net` | §8, §12 | NET-2/3 |
| Manifest / `BatchId` / interval slicing / synthetic corpus | `daemon-swarm-run` | §8, §6.3 | RUN-3 |
| `TrainerBackend` trait + `StubBackend` (the R↔E seam) | `daemon-swarm-run` | §5.1, §10.2, ABI §2.3 | (Wave-2 round loop) |
| Worker protocol types + CBOR codec | `daemon-swarm-run::protocol` | §10.2 | CLI-1 |
| Supervisor (spawn / respawn+backoff / meltdown) | `daemon-train-client` | §10.2, §13 | CLI-2/3 |

## Exported seams (freeze at Merge 1)

Public API surface other lanes and later waves consume. These signatures are the freeze contract.

### `daemon-swarm-net`

- `seam` module (MERGE-1 placeholders — see below): `ContentHash`, `RunId`, `RoundId`, `PeerId`,
  `PayloadKey`.
- `ControlPlane` trait — publish/subscribe of **already-signed** opaque control-message bytes;
  `ControlSubscription` (dedup-on-delivery receiver). `LoopbackGossip` implements it (in-process
  broadcast, fanout + content-hash dedupe).
- `PayloadStore` trait — `put` / `get` (hash-verify) / `head` (stat: size + hash) of opaque payload
  objects by `PayloadKey = (run, round, peer)`. `FsPayloadStore` implements it (rooted at a
  `daemon_core::ContainedRoot`), plus `prune(current_round)` retention and a typed
  `SwarmNetError::PayloadMiss` fed to the stall ladder.
- `ReceiptProducer` — polls a `PayloadStore` via `head` and emits `SignedReceipt { StorageReceipt,
  bytes }` (availability evidence as a signed message; the commit rule consumes only signed
  messages, §6.4 I6). Signing is injected via the `ReceiptSigner` seam.
- `ArtifactResolver` — scheme-dispatch (`ArtifactScheme::{File, R2, Hf, Https}`); only `File`
  wired this wave (blake3-verified). `R2`/`Hf`/`Https` are reserved variants that return
  `SwarmNetError::SchemeUnsupported` until the egress plane lands (reqwest is clippy-banned outside
  `daemon-egress`; no HTTP client is constructed this wave).

### `daemon-swarm-run`

- `data` module — `Manifest` / `ShardDesc` / `TokenWidth` (pre-tokenized shard format, §8),
  `Manifest::validate`, `Manifest::locate(BatchId) -> BatchLocation`, `slice_interval` (a peer's
  `BatchInterval` → `steps_per_round` × micro-batches), and `SyntheticCorpus` (deterministic seeded
  test corpus, u16 tokens).
- `backend` module — `TrainerBackend` trait (**the R↔E seam**, engine-agnostic: opaque bytes +
  plain structs, no burn/wasmtime) and `StubBackend` (deterministic xxh3 fake). See the verbatim
  signature below.
- `protocol` module — worker `Command` (down) / `Event` (up) + `encode`/`decode` CBOR codec,
  mirroring `daemon_infer::protocol` (length-framed over `daemon_provision::CutChannel`).

### `daemon-train-client`

- `TrainSupervisor` — lazy spawn, respawn-with-backoff, sliding-window crash-loop meltdown over the
  `daemon-train` worker binary; speaks `daemon_swarm_run::protocol` over a length-framed
  `CutChannel` (mirrors `LocalProvider` / `MettaCoprocessor`). `TrainClientConfig` tuning.

### `TrainerBackend` (verbatim — the seam to watch)

```rust
pub trait TrainerBackend: Send {
    type Error: std::error::Error + Send + Sync + 'static;
    fn build(&mut self, config: &[u8]) -> Result<(), Self::Error>;
    fn assess(&self, meta: &AssessMeta) -> Result<Assessment, Self::Error>;
    fn train_step(&mut self, batch: &BatchRef, ctx: StepCtx) -> Result<StepStats, Self::Error>;
    fn inner_update(&mut self, inner_step: u32) -> Result<(), Self::Error>;
    fn make_update(&mut self, round: RoundId) -> Result<Vec<u8>, Self::Error>;
    fn ingest(&mut self, round: RoundId, staged: &[StagedPayload]) -> Result<StateDigest, Self::Error>;
    fn checkpoint_save(&self) -> Result<Vec<u8>, Self::Error>;
    fn checkpoint_load(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;
}
```

## Placeholder types awaiting Merge 1 (proto swap)

Every one is behind a `seam` module and carries a `// MERGE-1: replace with daemon_swarm_proto::…`
comment, sized to be mechanically swappable (newtypes / plain structs, no behavior beyond
construction + hex/round-trip helpers). Merge 1 replaces the LOCAL definitions with the proto
crate's canonical types and deletes the `seam` re-exports.

| Placeholder (crate::path) | Replace with (proto) | Notes for Merge 1 |
|---|---|---|
| `daemon_swarm_net::seam::ContentHash` | `daemon_swarm_proto::ContentHash` | blake3-32; keep `of`/`to_hex`/`from_hex`. Consensus-critical: proto must be blake3 (not sha256, §6.4 delta). |
| `daemon_swarm_net::seam::RunId` | `daemon_swarm_proto::RunId` | opaque run identifier (currently a `String` newtype). |
| `daemon_swarm_net::seam::RoundId` | `daemon_swarm_proto::RoundId` | `u64` alias; watch for a proto newtype. |
| `daemon_swarm_net::seam::PeerId` | `daemon_swarm_proto::PeerId` | ed25519 **node** pubkey bytes (32) — never the iroh id (§7.2). Record-set ordering sorts by these bytes (§6.4). |
| `daemon_swarm_net::seam::PayloadKey` | (proto locator / key type) | `(run, round, peer)`; proto may fold this into `Commitment` locators. |
| `daemon_swarm_net::StorageReceipt` | `daemon_swarm_proto::StorageReceipt` | signed message shape `(run, round, peer, hash, size)` — the §6.4 CDDL message. `ReceiptSigner`/`SignedReceipt` become the real ed25519 signer + envelope. |
| `daemon_swarm_run::seam::BatchId` | proto `BatchId` (if proto owns it) | §6.3 assignment produces intervals; run only maps `BatchId → (shard, offset)`. |
| `daemon_swarm_run::protocol::{Command, Event, …}` | (stays in run per §10.1) | NOT a proto swap — this is the *worker* protocol (node↔worker), distinct from the *swarm* control protocol (proto). Keep in run; lane E (daemon-train) implements the worker side against it in Wave 3. |

## Things Merge 1 / later waves must watch for

- **Two distinct protocols.** `daemon_swarm_run::protocol` is the **worker** wire (node ↔
  `daemon-train` child, §10.2, CBOR over stdio). It is *not* the swarm control-plane protocol
  (`daemon-swarm.cddl`, lane P). Do not conflate them; only the `seam` id/hash types are shared.
- **`ContentHash` must land as blake3 in proto.** The whole integrity model (artifacts, payloads,
  checkpoints) is blake3 (§6.4 delta from Psyche's sha256). The net placeholder is already blake3;
  keep it so on swap.
- **Record-order staging is a consensus input.** `TrainerBackend::ingest` takes `staged` in
  caller-supplied order; the Wave-2 round loop MUST stage in `RoundRecord` order (sorted by node
  pubkey bytes, §6.4 I3). `StubBackend::ingest` is order-sensitive on purpose so a reordering bug
  is loud in tests.
- **`ControlPlane` carries already-signed bytes.** Signing/verification is lane P's envelope
  surface; net never signs control messages, only disseminates + dedupes them (§7.1: gossip is
  dissemination, never arbitration).
- **Egress is deferred.** `ArtifactResolver` dispatches on scheme but only `file://` is wired.
  `r2`/`hf`/`https` MUST route through `daemon_egress::EgressClient` when added (raw `reqwest` is
  clippy-banned workspace-wide). No HTTP client exists in lane R this wave.
- **fs goes through `ContainedRoot`.** `FsPayloadStore` is rooted at a `daemon_core::ContainedRoot`
  (openat2 RESOLVE_BENEATH|NO_SYMLINKS) so peer-supplied key components can never escape the store
  root. The `file://` artifact read is the one sanctioned raw-fs site (envelope-pinned URL,
  blake3-verified) with a documented `#[allow]`.
- **`daemon-train-client` depends on `daemon-swarm-run`** (for `protocol`), on top of the scaffold's
  `daemon-common` / `daemon-provision`. It still never links wasmtime / Burn (§10.1).
