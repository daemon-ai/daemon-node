# Swarm-training MVP — program ledger

Wave-0 scaffold coordination record for the daemon swarm-training MVP. This is the single source
of truth for the branch map, lane file-ownership, and the frozen-file rule. Lane agents: read this
before you touch anything.

## Base + branch map

- **Repo:** `daemon-node` (this is the Rust backend submodule; standalone checkout).
- **Base commit:** `0dbd720` (`0dbd7208826cdfafbc7214713ef38e7d2c51d621`,
  `merge(mirror/nv): WireVersion 39 — rungs 1+2+3 sealed (NV)`).
- **Trunk:** `integrations/swarm` — the integration branch. Wave-0 scaffold lands here (the commit
  list below). This is the merge target for every lane.
- **Lanes (branch off the Wave-0 scaffold tip — i.e. the commit that adds THIS file):**
  - `swarm/p1` — **P**rotocol / coordinator / observability lane.
  - `swarm/r1` — **R**untime / transport / node-supervisor lane.
  - `swarm/e1` — **E**ngine / tensor-ABI / guests lane.

All three lanes fork from the same HEAD and integrate back into `integrations/swarm`. Keep lanes on
disjoint file sets (table below) so merges are conflict-free by construction.

## Wave-0 commit list (on `integrations/swarm`, oldest → newest)

| Commit | Subject |
|---|---|
| `cc3df12` | `docs(specs): swarm training architecture + tensor ABI + TDD plan` |
| `de8fd64` | `build(deps): wasmtime + burn + blake3 + xxhash + ed25519 workspace pins` |
| `a621ca5` | `feat(swarm): crate scaffolds for the swarm training stack (spec §10.1)` |
| `26b08a5` | `build(nix): wasm32-unknown-unknown rust-std in devshell` |
| `31170e5` | `feat(xtask): build-guests + guests mini-workspace` |
| `53ddb21` | `build(deps): allow bincode unmaintained advisory (burn transitive)` |
| _(this file)_ | `docs(specs): swarm MVP program ledger` |

## Crate scaffolds (spec §10.1)

Nine empty-but-compiling crates, `crates/*/*`-globbed into the root workspace, each with
`[lints] workspace = true`, a spec-referencing crate doc, and a natural error type (no `todo!()`):

| Crate | Group | Deps (declared) | Lane |
|---|---|---|---|
| `daemon-swarm-proto` | `crates/contracts/` | serde, ciborium (wasm32-clean) | P |
| `det-core` | `crates/contracts/` | none (std only) | E |
| `daemon-train-sdk` | `crates/contracts/` | serde, ciborium | E |
| `daemon-swarm-net` | `crates/swarm/` | proto, tokio, reqwest | R |
| `daemon-swarm-run` | `crates/swarm/` | proto, net, tokio | R |
| `daemon-swarm-coordinator` | `crates/swarm/` | proto, axum, tokio | P |
| `daemon-swarm-observe` | `crates/swarm/` | proto, serde | P |
| `daemon-train` | `crates/coprocessor/` | proto, wasmtime, burn, blake3, xxhash-rust (+ bin) | E |
| `daemon-train-client` | `crates/coprocessor/` | daemon-common, daemon-provision, tokio | R |

## Dependency pins (root `[workspace.dependencies]`, resolved versions)

| Crate | Requirement | Resolved | Features |
|---|---|---|---|
| `wasmtime` | `46` | `46.0.1` | `default-features = false` + `runtime`, `cranelift`, `pooling-allocator` (fuel + epoch-interrupt are `Config` levers, no feature; no WASI) |
| `burn` | `0.21` | `0.21.0` | `default-features = false` + `std`, `ndarray`, `autodiff` (NO GPU backends) |
| `blake3` | `1` | `1.8.5` | default |
| `xxhash-rust` | `0.8` | `0.8.15` | `xxh3` |
| `ed25519-dalek` | `2` | `2.2.0` | default (already in-tree transitively; declared for the swarm lanes, wired in with envelope signing — lane P) |

`ciborium` (`0.2.2`) was already a workspace dep.

### deny.toml change

One documented advisory ignore added (licenses / bans / sources needed **no** changes):

- **`RUSTSEC-2025-0141`** (bincode unmaintained). `bincode 2.0.1` is an **unconditional** dep of
  `burn-core 0.21` (burn's record (de)serialization). It is an unmaintained-status advisory only
  (no CVE); the bincode team ceased development, so the advisory flags the crate itself — **no
  version pin or alternative resolves it** (the task's pin-over-ignore preference does not apply).
  burn is isolated to the `daemon-train` worker fault domain and never linked into the node process
  (§10.1). Matches the existing unmaintained-only ignores (paste / proc-macro-error2 / ttf-parser).
  Re-evaluate when burn moves off bincode.

## flake.nix change

The devShell toolchain now combines the pinned stable toolchain with
`fenix … targets.wasm32-unknown-unknown.stable.rust-std` (`rustToolchainDev` / `craneLibDev`),
scoped to the dev shell so package/build outputs keep the lean host-only toolchain. This is what
lets `xtask build-guests` cross-compile the guest modules in-shell.

## Lane file-ownership (disjoint; keep to your set)

| Lane | Owns (create / edit only within) |
|---|---|
| **P** (`swarm/p1`) | `crates/contracts/daemon-swarm-proto/`, `crates/swarm/daemon-swarm-coordinator/`, `crates/swarm/daemon-swarm-observe/`, `daemon-swarm.cddl` (new, repo root or the api crate per §10.4 authoring rules) |
| **R** (`swarm/r1`) | `crates/swarm/daemon-swarm-net/`, `crates/swarm/daemon-swarm-run/`, `crates/coprocessor/daemon-train-client/`, `tests/daemon-swarm-e2e/` (new), `bins/` |
| **E** (`swarm/e1`) | `crates/contracts/det-core/`, `crates/contracts/daemon-train-sdk/`, `crates/coprocessor/daemon-train/`, `guests/`, the `xtask build-guests` subcommand |

Cross-lane dependency edges are already wired via `[workspace.dependencies]` path entries (a lane
consuming another lane's crate uses `{ workspace = true }` and does **not** edit that crate).

## FROZEN files — single-writer rule (non-negotiable)

After the Wave-0 scaffold (this ledger commit), the following are **FROZEN**. Lane agents MUST NOT
modify them; a change here would collide across all three lanes and break the disjoint-merge
guarantee. Route any needed change through the integration owner as a separate, coordinated commit
on `integrations/swarm`.

- **`Cargo.toml`** (root) — workspace members glob, `exclude = ["guests"]`, `[workspace.dependencies]`, `[workspace.lints]`, profiles.
- **`deny.toml`** — advisory/license/ban/source policy.
- **`flake.nix`** — devShell toolchain + targets.

Adding a **new member crate** to a lane is fine (the `crates/*/*` glob picks it up with no root
edit). Adding a **new third-party dependency** requires a root `Cargo.toml` change → it is NOT a
lane action; request it from the integration owner (who also re-runs `cargo deny check`).

## Notes for lane agents (not obvious from the code)

- **Gates (from the worktree root, all currently green):** `cargo fmt --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, `cargo deny check`,
  `cargo test --workspace`, `cargo run -p xtask -- build-guests`, `typos docs/specs/`. Run
  everything via `nix develop --command …`.
- **reqwest is banned (clippy `disallowed_types`).** `daemon-swarm-net` declares `reqwest` for the
  egress plane, but a raw `reqwest::Client`/`ClientBuilder` fails the clippy gate workspace-wide
  (see `clippy.toml`). Route outbound HTTP through `daemon_egress::EgressClient`. No client is
  constructed in the scaffold.
- **fs / process / env bans** also live in `clippy.toml` (`daemon_core::ContainedRoot`,
  `daemon_provision`/`daemon-processes` for spawns, `EnvPolicy` for child env). `xtask` is
  `#[allow(clippy::disallowed_methods)]` crate-wide, which is why `build-guests` may call
  `Command::new("cargo")` directly.
- **Heavy trees (`wasmtime`, `burn`) build in the default workspace gate here** because
  `daemon-train` declares them directly (scaffold). In the shipped product they move to out-of-gate
  per-backend Nix lanes (§10.1); lane E should preserve that intent (feature-gate / lane-split the
  worker) rather than leaving burn/wasmtime on the default path forever.
- **No GPU backends** are in the graph — burn is `ndarray + autodiff` only. A stray `wgpu-*` set
  sits in `Cargo.lock` as an unreachable orphan (not compiled, not seen by `cargo deny`); do not
  "clean it up" by enabling a wgpu feature.
- **`guests/` is a SEPARATE workspace** (`exclude = ["guests"]`). It links `daemon-train-sdk` by
  path (`../../crates/contracts/daemon-train-sdk`) and builds only via `xtask build-guests` for
  `wasm32-unknown-unknown`. `guests/target/`, `guests/Cargo.lock`, and `*.wasm` are gitignored (lane
  E may choose to commit `guests/Cargo.lock` later for reproducible artifacts).
- **wasm32 rust-std is only in the dev shell.** A bare `cargo build --target
  wasm32-unknown-unknown` outside `nix develop` will fail — always use the dev shell.
- **`daemon-swarm.cddl` does not exist yet** — lane P creates it and (per §10.4) wires a swarm
  parity check; if you extend the `xtask cddl` gate for it, that xtask edit is shared tooling, not a
  frozen-file change, but coordinate it.
- Contracts crates that must stay dependency-lean (`daemon-swarm-proto` serde+ciborium,
  `det-core` std-only, `daemon-train-sdk` serde+ciborium) hand-roll their error types
  (`std::error::Error`) instead of using `thiserror`. Keep them lean — `daemon-swarm-proto` and
  `daemon-train-sdk` are on the `wasm32` path.

## Merge 1 — frozen interfaces

Wave-1 lanes P1 (`swarm/p1` @ `3c60271`), R1 (`swarm/r1` @ `806b926`), and E1 (`swarm/e1` @
`73c7a68`) are merged into `integrations/swarm` (three `--no-ff` merges, lane history preserved).
The seams below are **frozen**: Wave-2+ extends them **additively** only — any breaking change needs
a Merge-coordination note here. All gates green on the merged trunk (see the gate list in "Notes for
lane agents"; plus `cargo test -p daemon-train-sdk --features sim`, `cargo build --target
wasm32-unknown-unknown -p daemon-swarm-proto`, and `cargo run -p xtask -- build-guests`).

### The two protocols (do not conflate)

- **Swarm control protocol** — `daemon-swarm-proto` (lane P). The signed, consensus-critical wire:
  canonical CBOR + the `SignedMessage` frame over `daemon-swarm.cddl`. Peer↔coordinator, peer↔peer.
  Versioned by `SwarmProtoVersion`.
- **Worker protocol** — `daemon_swarm_run::protocol` (lane R). The node↔`daemon-train` **child**
  wire (CBOR over a length-framed `daemon_provision::CutChannel` stdio cut, §10.2). NOT a swarm wire;
  only the shared id/hash types (`PeerId`, blake3 `Hash`) are common. Stays in `daemon-swarm-run`;
  lane E's worker implements the child side in Wave 3.

### `daemon-swarm-proto` API (the single authority for wire shapes; wasm32-clean)

- Canonical codec: `to_canonical_vec<T: Serialize>`, `from_canonical_slice<T: DeserializeOwned>`
  (RFC 8949 §4.2 deterministic CBOR — the bit-identity seam; never fork a second encoder).
- Byte newtypes (all CBOR `bstr`): `PeerId`/`Hash`/`Root`/`Seed`(32), `Signature`(64), `IrohId`(32),
  `StateDigest`(16). `blake3_hash(&[u8]) -> Hash`. Ordered lexicographically by bytes.
- Signing (ed25519 over canonical CBOR, deterministic/no-RNG): `SigningKey`/`VerifyingKey`,
  `peer_id`, `sign_canonical`/`verify_canonical`, `Signed<T>` (`seal`/`verify`).
- Envelope: `Envelope` (+ `RunSection`/`ExperimentSection`/`Artifact`/`DataSection`/`Requirements`/
  `Phases`), `validate`, `freeze(&SigningKey) -> FrozenEnvelope`, `FrozenEnvelope::{verify, bytes,
  hash, config_bytes, signature, signer}`, `ENVELOPE_SCHEMA_MAJOR`.
- Capability: `Capability { name, version }`, `CapabilitySet::admits`.
- Set commitment (blake3 merkle over `(peer, hash)` **sorted by peer pubkey bytes**, §6.4 I3):
  `commit_set`, `SetCommitment { root, count }`, `SetCommitmentTree`, `MembershipProof`,
  `verify_membership`.
- Messages (the 7 round msgs + Join/Heartbeat): `RoundOpen`, `Commitment`, `Attestation`,
  `StorageReceipt` (`{ round, verified: Vec<RecordEntry{peer,hash,size}> }`), `RoundRecord`,
  `Digest`, `Straggle`, `Join`, `Heartbeat`; enum `SwarmMessage`; frame `SignedMessage { version,
  payload, signer, sig }` with `sign`/`verify`/`verify_for_run`. Plus `Locator`, `BatchWindow`,
  `AttestEntry`, `RecordEntry`, `ThroughputClass`, `StraggleStatus`.
- Version: `SwarmProtoVersion(u16)`, `SWARM_PROTO_VERSION` (= `1`), `accepts`/`check_join`
  (exact-match join gate).
- Digest: `StateLayout`, `DigestSchedule`, `derive_schedule`, `digest_state` (seed-keyed xxh3-128).

### `daemon-swarm-net` — `SwarmTransport` seams (lane R)

- `ControlPlane` (`publish(&[u8])` of already-signed bytes, `subscribe() -> ControlSubscription`;
  content-hash dedupe). `LoopbackGossip` impl. `ControlSubscription::{recv, try_recv}`.
- `PayloadStore` (`put`/`get`(hash-verify)/`head`(`PayloadStat { hash: Hash, size }`)) keyed by
  `PayloadKey = (RunId, RoundId, PeerId)`. `FsPayloadStore` impl (rooted at `ContainedRoot`,
  `prune(run, current_round)` retention, typed `PayloadMiss`).
- `ReceiptProducer<S>` — polls a `PayloadStore` and emits proto `SignedMessage`s carrying a
  `StorageReceipt` (`produce(&key)` single-entry; `produce_round(round, keys)` aggregated). Signs
  with an injected ed25519 `SigningKey` + pinned `SwarmProtoVersion`.
- `ArtifactResolver` — `ArtifactScheme::{File, R2, Hf, Https}`; only `File` wired (blake3-verified).
  `R2`/`Hf`/`Https` return `SwarmNetError::SchemeUnsupported` pending `daemon-egress`.
- Shared vocabulary (`seam`): `ContentHash` = proto `Hash`, `PeerId` = proto `PeerId`, `RoundId`
  (`u64`), and the local-only `RunId`/`PayloadKey` re-expressed over the proto primitives.

### `daemon-swarm-run` — `TrainerBackend` (the R↔E seam) + data pipeline (lane R)

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

`ingest` consumes `staged` in caller order; the round loop MUST stage in `RoundRecord` order —
sorted by node pubkey bytes (§6.4 I3), the **same key proto's `commit_set` uses**. Cross-crate test:
`daemon-swarm-run/tests/record_ordering.rs` (`staging_order_matches_proto_commit_set`).
`StubBackend` is the deterministic Wave-1 fake. Data pipeline: `Manifest`/`ShardDesc`/`TokenWidth`,
`Manifest::{validate, locate, total_*}`, `slice_interval`, `SyntheticCorpus`, `BatchId` (`u64`).

### `daemon-swarm-run::protocol` — worker protocol message set (frozen, node↔worker)

- `Command`: `AssessRun`, `JoinRun`, `Throttle`, `Leave`.
- `Event`: `Ready`, `Probed(Hardware)`, `Assessed(Eligibility)`, `RunPhase`, `RoundProgress`,
  `RoundOutcome`, `Metric`, `CheckpointPublished`, `Warning`, `Error`.
- Codec: `encode<T: Serialize>`/`decode<T: DeserializeOwned>` (CBOR body; the `u32` length prefix is
  the `CutChannel`'s). Consumed by `daemon-train-client::TrainSupervisor` (spawn/respawn/meltdown).

### `det-core` kernel signatures (fixed-order fp32; lane E)

```rust
det_sum(&[&[f32]]) -> Result<Vec<f32>, DetError>;   det_l2norm(&[f32]) -> f32;
det_axpy(&mut [f32], f64, &[f32]) -> Result<(), DetError>;   det_scale(&[f32], f64) -> Vec<f32>;
det_add / det_sub / det_mul (&[f32], &[f32]) -> Result<Vec<f32>, DetError>;   det_sign(&[f32]) -> Vec<f32>;
det_chunk_scatter_add(&mut [f32], &[f32], &[u32], usize) -> Result<(), DetError>;
det_chunk_scatter(&[f32], &[u32], usize, usize) -> Result<Vec<f32>, DetError>;
det_absmax_unpack(&[u8], usize, u32) -> Result<Vec<f32>, DetError>;
```

`f64` scalars cast to `f32` **inside** the kernel (one shared cast site, ABI §5.9). `det_absmax_unpack`
layout frozen (ABI §6.6): per-chunk LE `f16` absmax then `chunk` codes of `bits` width, LSB-first,
chunk-major, byte-padded; symmetric linear codebook.

### `tabi@1` subset — the frozen 50-import vocabulary (lane E)

Host `Linker` and the SDK extern block agree name-for-name; `daemon-train/src/phase.rs` is the
normative phase-legality table (frozen with these names). Later waves add the remaining 108-import
`tabi@1` vocabulary **additively** (§9).

```
param, persistent, det_persistent, drop, param_round_base, backward, grad, zero_grads, assign,
zeros, ones, full, add, sub, mul, mul_s, matmul, relu, cross_entropy, scalar, metric, log,
abi_minor, adamw_step, batch_tokens, batch_size, batch_seq_len, upd_new, upd_push_bytes,
upd_push_tensor, upd_sections, upd_kind, upd_bytes_len, upd_read_bytes, upd_tensor, det_zeros,
det_sum, det_scale, det_l2norm, det_sign, det_add, det_sub, det_mul, det_absmax_unpack,
det_chunk_scatter_add, det_chunk_scatter, det_assign, det_param, det_reset_param_to_base,
det_axpy_param
```

SDK surface (frozen): `Experiment` trait (`manifest`/`build`/`step`/`inner_update`/`make_update`/
`ingest`) + `experiment!` macro (→ `da_abi`/`da_manifest`/`da_defaults`/`da_alloc`/`da_free` +
`da_build`/`da_step`/`da_inner_update`/`da_make_update`/`da_ingest_updates`); wrapper types
`Tensor`/`DetTensor`/`Param`/`Persistent`/`DetPersistent`/`Batch`/`StepCtx`/`Config`/`Manifest`/
`UpdateBuilder`/`UpdatesView`; `--features sim` → in-crate CPU backend over `det-core`.
`da_manifest`/`da_defaults` return the CBOR `(ptr, len)` packed as one `u64` (`ptr << 32 | len`),
not wasm multi-value (E1 note). `daemon-train` host: `Worker::{new, load_module, instantiate}` +
the arena/trap-taxonomy/phase-table/budgets, with `OpBackend`/`CpuBackend`.

### Seam-swap deviations (recorded)

Merge 1 swapped R1's `seam` placeholders + receipt types for proto per plan, with these
lane-report deltas (none behavioral, none consensus-affecting):

- **`ContentHash` → proto `Hash` by alias.** `daemon_swarm_net::seam::ContentHash` is now
  `pub use daemon_swarm_proto::Hash as ContentHash` (kept the descriptive net-local name; the type
  is proto's). Call sites that used `ContentHash::of(b)` now use `daemon_swarm_proto::blake3_hash(b)`
  (proto has no `Hash::of`). Wire encoding of a hash changed array-of-uint → `bstr` (proto's form),
  which is what everything crossing a signature/wire now uses.
- **`Hash` has no `from_hex`.** Proto exposes `to_hex` but no inverse. The one consumer
  (`daemon-swarm-run` manifest validation of the hex `ShardDesc.blake3` string field) uses a local
  `is_blake3_hex` predicate expressed over `Hash::LEN` rather than parsing a `Hash`. The manifest
  hash stays a JSON hex `String` (not a typed `Hash`), unchanged.
- **`RunId`/`RoundId`/`PayloadKey`/`BatchId` kept local.** Proto keeps run ids as `String`
  (`Join::run_id`) and rounds/batch ids as bare `u64` (`RoundOpen::round`, `BatchWindow`), so there
  is no proto newtype to swap for. `RoundId`/`BatchId` are `u64` aliases; `RunId`/`PayloadKey` are
  local newtypes/structs keyed over the proto `PeerId`. R1's `seam` re-exports are retained as the
  shared vocabulary module (not deleted) since they now resolve to proto types + these locals.
- **Receipts re-expressed as proto `SignedMessage`.** R1's local `StorageReceipt`/`SignedReceipt`/
  `ReceiptSigner`/`UnsignedSigner` are gone. `ReceiptProducer` now emits proto
  `SwarmMessage::StorageReceipt` inside a real ed25519 `SignedMessage` (via `SignedMessage::sign`).
  Shape shift: proto's `StorageReceipt` is round-scoped with a `verified: Vec<RecordEntry>` batch and
  carries **no `run` field** (run is contextual — it stays in the transport `PayloadKey` for store
  lookup). Per-key `produce` yields a single-entry receipt; `produce_available` was replaced by
  `produce_round(round, keys)` which aggregates all available keys into one signed message.
- **No root `Cargo.toml`/`deny.toml`/`flake.nix` change was needed.** The proto path dep was already
  wired for `daemon-swarm-net`/`daemon-swarm-run` (`daemon-swarm-proto = { workspace = true }`), so
  the swap added no third-party dependency and required no `cargo deny` re-run beyond the gate. (Net
  keeps `blake3` — still used directly for gossip dedupe; net's now-unused `ciborium` and run's
  now-unused direct `blake3` are left declared, harmless to the gates, flagged for `audit-cleanup`.)

### Wave-2 must know

- **`burn` is still on the default gate** (declared directly by `daemon-train`; a full
  `cargo test/clippy --workspace` cold-builds wasmtime+burn). The `OpBackend` trait is the
  **one-crate seam** that makes lane-splitting burn/CubeCL off the default path a single-crate change
  — do it in Wave 2. Do NOT change the `tabi@1` import names or the phase-legality table doing so.
- **Egress schemes unsupported.** `ArtifactResolver` dispatches `r2`/`hf`/`https` but returns
  `SchemeUnsupported` until `daemon_egress::EgressClient` is wired (raw `reqwest` is clippy-banned
  workspace-wide; `daemon-swarm-net` declares `reqwest` but constructs no client). Wire egress before
  any non-`file://` artifact/payload plane.
- **Guest `.wasm` location.** Guests build into `guests/target/wasm32-unknown-unknown/release/`
  (gitignored, separate workspace); `daemon-train`'s guest-lifecycle tests locate via
  `SWARM_TEST_GUEST_DIR` else the manifest-relative path, building on demand if absent (needs the
  dev-shell `wasm32-unknown-unknown` rust-std). Verified from the integration worktree at Merge 1.
- **Additive-only extension.** The proto API, `SwarmTransport` traits, `TrainerBackend`, worker
  protocol, `tabi@1` subset, det-core signatures, and the phase table are frozen; extend, do not
  break.

## Merge 2 — P0 milestone (real coordinator drives the e2e)

Wave-2 lanes P2 (`swarm/p2` @ `2032ade`), R2 (`swarm/r2` @ `1358f36`), and E2 (`swarm/e2` @
`e32b047`) are merged into `integrations/swarm` (three `--no-ff` merges, lane history preserved,
`swarm/p2` → `swarm/r2` → `swarm/e2`). File sets were disjoint by construction; the only textual
overlap was `Cargo.lock` (auto-merged, settled by `cargo check`). Lane ledgers are separate files
(`swarm-ledger-{p2,r2,e2}.md`); no lane touched this program ledger or `guests/` member lists.

**The P0 milestone landed:** the `tests/daemon-swarm-e2e` end-to-end now runs 3 peers × 20 rounds
over `StubBackend` driven by the **real** `daemon_swarm_coordinator::tick` — R2's TEST-ONLY
`ScriptedCoordinator` is gone.

### What was swapped

- **`ScriptedCoordinator` → `TickCoordinator`** (`daemon-swarm-run/src/harness.rs`, `harness`
  feature). The pure `tick` stays in `daemon-swarm-coordinator`; the harness is the **impure shell**:
  it holds a `CoordinatorState`, signs the coordinator's *unsigned* `RoundOpen`/`RoundRecord`
  `Output::Publish` values with the coordinator identity and broadcasts them over `LoopbackGossip`,
  and feeds `tick` the inbound signed peer messages, synthesized `Join`s (bootstrap roster),
  `StorageReceipt` availability evidence, and scripted `Clock` inputs. `daemon-swarm-run` gains an
  **optional** `daemon-swarm-coordinator` dep gated on the `harness` feature (+ a dev-dep for its own
  `cfg(test)` harness tests) — a lane-owned `Cargo.toml` edit; the frozen root `Cargo.toml` already
  declared the path dep, so **no root/deny/flake change and no `cargo deny` re-run** were needed.
- **`engine.rs` `assignment::interval_for`** — equal-split replaced with P2's throughput-weighted
  `daemon_swarm_proto::assignment::assign_batches` (all StubBackend peers `ThroughputClass::C1`,
  `overlap_bps = 0` exact partition). The per-peer→interval mapping is now seed-shuffled; the
  transcript changed, agreement held.

### Impedance mismatches found (P2 tick I/O ↔ R2 harness) — the valuable findings

1. **The commit rule needs availability *evidence*; R2 peers don't self-attest.** P2's `tick`
   admits a payload only with a `StorageReceipt` **or** a witness-quorum of `Attestation`s (§6.4 I6).
   R2's `RoundEngine` never attests its **own** payload (a peer's self-prefetch short-circuits in
   `prefetch`), and a straggler cannot attest the peer whose object it failed to fetch. So
   witness-quorum coverage alone **cannot evidence every payload** at small/stalled rosters: a single
   peer (the RUN-5 barrier test) or peer0's object in the stall round (peer1 can't fetch it, so only
   peer2 attests it → 1 < quorum(3)=2) would **never finalize**. Resolution: the harness shell runs
   the **coordinator-as-storage-client** `StorageReceipt` path — on each `Commitment` it `HEAD`s the
   shared store and feeds a signed `StorageReceipt`, exactly P2's intended primary evidence path
   ("the coordinator's HEADs already arrived as signed StorageReceipt inputs", commit.rs). Evidence
   is thereby decoupled from peer fetch success. **This is the single most important integration
   finding** — the witness-attestation path is insufficient with the current peer engine.
2. **No roster without `Join`s; the engine sends none.** `RoundEngine` subscribes and waits for
   `RoundOpen`; it never `Join`s. The shell synthesizes each peer's signed `Join` (it re-derives the
   deterministic peer keys) and feeds them at bootstrap so the roster forms through the real
   admission path, then clocks past warmup to open round 0.
3. **Warmup is timeout-only (P2 note).** The shell supplies two bootstrap `Clock` inputs
   (`WaitingForMembers → Warmup → RoundTrain`); there is no per-peer model-ready confirmation this
   wave.
4. **Deterministic finalization without wall-clock coupling.** `tick` finalizes a round
   *event-driven* the moment it is fully committed + evidenced (no clock) — the happy/catch-up path
   thus needs **zero** clocks and is byte-identical across runs. A round blocked by a straggler that
   won't commit is forced by a single timeout `Clock` **only once every healthy peer is accounted**
   (committed+receipted, or `Straggle(Stalled)` this round) — the same content-driven rule R2's
   scripted coordinator used; a generous quiescence guard covers a peer gone fully silent (left).
   Neither ever fires on the happy path. `RoundOpen.deadline_unix_s` varies with the injected clock
   but is not consumed by peers and never enters a digest.
5. **Attestations are now redundant but harmless.** Peers still publish `Attestation`s; the shell
   feeds them (all peers are witnesses via `witness_target = 0`), they populate `RoundState` and are
   accepted, but receipts carry the actual evidence.

### P0 evidence (e2e results)

- `twenty_rounds_all_agree_with_stall_and_catchup`: **20 rounds**, 3 peers, **all digests equal
  every round** (`all_agree`, 3 digests/round × 20); peer 1 **straggles round 7** (injected 2-`get`
  payload miss of peer 0's object) and **catches up round 7 at round 8 open** (`Straggling{7}` +
  `CaughtUp{7}`), no peer leaves.
- `digest_transcript_is_byte_identical_across_runs`: two runs → **identical 20-entry agreed
  transcript** (determinism, incl. the stall path).
- **Replay assertion (PROTO-20 spirit):** the shell records the exact `tick` input sequence + a
  canonical-CBOR `CoordinatorState` snapshot after each `RoundRecord`; `CoordinatorReplay::verify`
  re-runs `tick` over the recorded inputs and asserts a **byte-identical per-round state trajectory**
  (20 snapshots). Green, and stable across repeated runs.
- All 29 `daemon-swarm-run` tests + 2 `record_ordering` (I3) tests pass against the real coordinator,
  including the single-peer barrier test (RUN-5) and the leave test (`stall_budget_exhausted_leaves`).

### Frozen Wave-2 surfaces (freeze at Merge 2; extend additively only)

- **Assignment (`daemon_swarm_proto::assignment`, wasm32-clean):** `Lcg`/`seeded_lcg`,
  `deterministic_shuffle`, `witness_quorum`, `class_weight`, `select_committee`/`select_verifiers`/
  `elect_checkpointer`, `global_batch_at`/`advance_cursor`, `assign_batches`, `Committee`,
  `WITNESS_TARGET_DEFAULT`. Golden vectors pinned (daemon seed `0xDAE07E57`).
- **Coordinator (`daemon-swarm-coordinator`, pure library, wasm32-clean):**
  `tick(CoordinatorState, Input) -> (CoordinatorState, Vec<Output>)`; `admit`;
  `Input`/`Output`/`Notice`/`Rejection`/`AdmissionReject`; `Phase`; `CoordinatorState`
  (canonical-CBOR-serializable, ring of `NUM_STORED_ROUNDS=4`); `RunConfig`/`CoordinatorParams`
  (`from_envelope`); `ready_to_update_epoch`. **`tick` emits UNSIGNED coordinator messages — the
  driver signs them.**
- **RoundEngine (`daemon_swarm_run::engine`):** `RoundEngine::{new, run}`, `EngineConfig`,
  `EngineEvent`, `RunOutcome`; `verify_record_set`. Peer-side barrier (I2), record-order staging
  (I3), stall ladder. Checkpoint types (`daemon_swarm_run::checkpoint`): `CheckpointManifest`,
  `save/load_checkpoint`, `resync_by_replay`, `ReplayStep`. Harness seam
  (`daemon_swarm_run::harness`, `feature = "harness"`): `run_swarm`/`run_swarm_with`, `SwarmConfig`,
  `SwarmRun` (+ `CoordinatorReplay`), `StallFault`/`FaultyStore`.
- **`tabi@1` = 66 ops** (`daemon_train_sdk::TABI_IMPORTS`, pinned by `daemon-train/tests/abi_surface.rs`;
  host `Linker` + SDK extern + `phase.rs` table agree name-for-name, all three length 66).
- **Profile config schemas (`daemon_train_sdk::profiles`):** `SparseLocoCfg`, `DiLoCoCfg`, `DemoCfg`;
  `TinyLlamaCfg` (guests/tiny-llama). det-core compression kernels (`dct2`/`idct2`/`topk_chunk`/
  `absmax_pack`) additive to the frozen signatures.

### Remaining `MERGE-2` markers — all genuinely Wave-3 (verified)

| Site | Deferred work | Why Wave-3 |
|---|---|---|
| `engine.rs` (fetch-overlap note) | dedicated in-peer concurrent fetch task | buys nothing over the fast `FsPayloadStore` + would add digest nondeterminism; worthwhile only once the real **iroh/r2 payload plane** lands |
| `engine.rs` `verify_record_set` | fetch `record-set.cbor` via `set_locator` at large rosters | the MVP small rosters ride the **inline** set; the object-fetch path needs the r2/iroh plane |
| `checkpoint.rs` `resync` | wire the consensus/quorum-digest **desync trigger** | needs `daemon-swarm-observe` / the coordinator's digest tally (**observe-driven desync trigger**); the replay fold itself is done |
| `net/fetch.rs` | alternate `BlobTicket` locators / network fetch fallback | awaits the **iroh-blobs** plane |

(The `harness.rs` and `engine.rs` assignment "MERGE-2 resolved" comments are annotations that the
scripted-coordinator swap + assignment swap are **done**, not pending work.)

### Wave-3 must know (carried from lane reports + this integration)

- **Peer self-attestation gap.** Because the peer engine does not attest its own payload, the
  witness-quorum evidence path alone under-covers small/stalled rosters. Wave 3 either (a) has peers
  self-attest, (b) keeps the coordinator `StorageReceipt` producer as the primary evidence source
  (the runnable coordinator in `bins/` should ship one), or (c) both. The e2e proves (b) works.
- **Envelope-hash admission** is structural (`JoinCandidate::asserted_hash` + `EnvelopeHashMismatch`)
  but `tick` passes `None` — the frozen `Join` carries no hash. Enforce once `Join` gains an additive
  `envelope_hash` field / assessment token.
- **Warmup timeout-only** (no per-peer model-ready); **root-only attestation coverage** (membership
  proofs, not inline) is a Wave-3 add; **verifier committee** is a no-op at `verification_percent=0`.
- **`burn` is still on the default gate** (declared by `daemon-train`); the `OpBackend` seam still
  stands — lane-split burn/CubeCL off the default path is outstanding Wave-2/3 lane-E work.
- **E2 ↔ R2 are not wired together** (TinyLlama/profiles + the real wasm host vs the RoundEngine):
  verified **no accidental coupling** — `daemon-swarm-run` has no dep on `daemon-train`/`-sdk` and
  vice-versa. Real-backend wiring (the `TrainerBackend` impl over the wasm host) is Wave-3 work.
- **Egress schemes** (`r2`/`hf`/`https`) still return `SchemeUnsupported` pending
  `daemon_egress::EgressClient`.
