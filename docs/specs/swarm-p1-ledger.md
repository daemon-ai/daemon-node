# Swarm P1 + Transport — program ledger

Wave-0 scaffold coordination record for the **Swarm P1 + Transport** program (Workstream A: node
wire + burn-wgpu GPU training to the P1 "160M on 1 GPU" gate; Workstream B: real transport planes —
R2 via the daemon-cloud `apps/swarm` worker, self-hosted iroh gossip, live worker transport). This
is the single source of truth for the trunk, the lane file-ownership, the frozen-file / frozen-
interface rule, and the two reference packs. Lane agents working in this worktree: **read this
before you touch anything** — it carries everything you need without reaching into `~/.cursor`.

The MVP-era scaffold record is `swarm-mvp-ledger.md` (its Merge 1/2/3 "frozen interfaces" sections
remain authoritative for the seams frozen at the MVP). This ledger governs the P1 program on top of
that MVP.

## Base + trunk

- **Repo:** `daemon-node` (Rust backend submodule; standalone checkout).
- **Base commit:** `e2e08c3` (`mirror(merge-3): MVP gate — final program ledger`) — the completed
  swarm-training MVP on `integrations/swarm`.
- **Trunk:** `integrations/swarm-p1` (one shared trunk, forked from `e2e08c3`). A/B lanes are
  interleaved wave-wise and merge back here; the integration owner owns the frozen files, the
  `WireVersion` bump, seam swaps, and this ledger.
- **Worktrees:** each lane subagent works EXCLUSIVELY in its assigned worktree under
  `/home/j/experiments/daemon-worktree/` (daemon-node) or a branch checkout of
  `/home/j/experiments/daemon-cloud/daemon-api` (BC lane). Never modify the main checkouts, never
  `git push`, never `--no-verify`.

## Lane ownership table (disjoint by construction)

| Lane | Wave | Owns (disjoint) |
|---|---|---|
| **W1** wire/node | 1 | `crates/contracts/daemon-api/*` + CDDL, `bins/daemon/src/config.rs`, new node swarm-service module, `swarm.db` migrations, superproject `justfile` recipe (proposed as a diff, not committed) |
| **G1→G2** GPU | 1→2 | `crates/coprocessor/daemon-train/src/{backend.rs,burn_backend.rs,wasm_backend.rs,meta.rs}`, worker `backend` module |
| **B1** store/egress | 1 | `crates/swarm/daemon-swarm-net/src/{r2_store.rs,presign.rs,artifact.rs,fetch.rs}` |
| **M1→M2** model/data | 2→3 | `crates/contracts/daemon-train-sdk/src/models.rs`, `guests/*`, `crates/swarm/daemon-swarm-run/src/data.rs` (additive), `xtask` corpus subcommand, parity harness in `tests/` |
| **B2** gossip | 2 | `crates/swarm/daemon-swarm-net/src/iroh_gossip.rs` (+ relay setup docs/scripts) |
| **B3** live transport | 3 | `crates/swarm/daemon-swarm-run/src/{engine.rs,local_coordinator.rs}` (additive), `bins/swarm-local`, `daemon-train-client`, worker `transport` module, `tests/daemon-swarm-e2e` |
| **BC** coordinator app | 3 | daemon-cloud `apps/swarm/*`, `apps/gateway/src/routes/swarm.ts` + service binding, `packages/shared/src/swarm/` |
| **Integration owner** | merges | frozen files, WireVersion bump, seam swaps, ledger |

Cross-lane dependency edges are wired via `[workspace.dependencies]` path entries: a lane consuming
another lane's crate uses `{ workspace = true }` and does **not** edit that crate. Adding a **new
member crate** to a lane is fine (the `crates/*/*` glob picks it up with no root edit). Adding a
**new third-party dependency** requires a root `Cargo.toml` change → that is NOT a lane action;
request it from the integration owner (who also re-runs `cargo deny check`).

## FROZEN files — single-writer rule (integration owner only)

After this Wave-0 scaffold, the following are **FROZEN**. Lane agents MUST NOT modify them; a change
here collides across lanes and breaks the disjoint-merge guarantee. Route any needed change through
the integration owner as a separate, coordinated commit on the trunk.

- **`Cargo.toml`** (root) — workspace members glob, `exclude = ["guests"]`, `[workspace.dependencies]`, `[workspace.lints]`, profiles.
- **`deny.toml`** — advisory/license/ban/source policy.
- **`flake.nix`** — devShell toolchain + targets + package/devShell lanes.

## FROZEN interfaces (extend additively only)

From the MVP (inventory + exact shapes in `swarm-mvp-ledger.md` Merge 1/2/3): proto wire + envelope
(`daemon-swarm-proto`), coordinator `tick`, `SwarmTransport` (`ControlPlane` / `PayloadStore`),
`TrainerBackend`, worker `protocol::{Command,Event}`, `det-core` signatures, `OpBackend` (defaults-
based additions OK), the `tabi@1` **66-op** list + `phase.rs` table (additive growth allowed **only
until the P1 exit gate**, spec §16, then frozen forever). Wave-2 froze the assignment module, the
pure coordinator library, `RoundEngine`/harness, profile config schemas, and the 66-op vocabulary.

New P1 seams to be frozen at their merges (see the wave/gate structure): `SwarmApi` wire surface +
`PresignClient` + tolerance-class harness API (Merge 1); `IrohGossip` config surface + manifest/
tokenizer conventions + 160M preset schema (Merge 2); the presign/WS HTTP surface between daemon-
node and daemon-api (frozen early via shared DTO fixtures — see Risks).

## Determinism story (the P1 relaxation — spec §7.2 sanctions it)

The MVP's cross-peer bit-identity comes from `CpuBackend`'s fixed-order fp32 arithmetic (two
`CpuBackend`s over identical inputs produce byte-identical digests). Under burn/GPU that no longer
holds:

- **Native lane becomes a tolerance class.** burn's autodiff (and any GPU backend's kernels) are not
  bit-wise equal to the CPU tape, so the native training path (params, grads, losses) is compared
  against `CpuBackend` under **per-op rtol/atol tolerance classes** (G1 builds the machinery, HOST-3),
  not exact equality. This is expected and spec-sanctioned (§7.2: consensus is over the **det lane**,
  not the native math).
- **Bit-exactness remains det-lane-only.** The `det_*` ops stay on det-core CPU fp32 with device→host
  materialization at the ingest boundary (masters fp32 host-side, requantize on
  `det_axpy_param`/`det_reset_param_to_base`, ABI §5.9 residency contract). The consensus digest
  (`digest_state`, seed-keyed xxh3-128 over the det lane) is therefore **backend-independent** and
  MUST stay bit-identical across `CpuBackend` / `BurnBackend(ndarray)` / `BurnBackend(wgpu)`.
- **The guard tripwire:** the cross-backend det-digest equality test (G1's HOST-7 extension: a
  `CpuBackend` run and a `BurnBackend(ndarray)` run produce equal det-lane digests; G2 extends it to
  wgpu). A det-lane residency mistake on GPU desyncs peers and is caught late by digests — this test
  is the early tripwire. See Risks 1–2.

## Wave / gate structure

- **Wave 0 — scaffold (this):** trunk + worktrees; this ledger; the batched frozen-file pass (burn
  `wgpu`/backend features, iroh 0.97 pins behind a `daemon-swarm-net/iroh` feature, deny audit, flake
  lanes); the worker-binary module split; `EgressClient::put`. Gates green before lane launch.
- **Wave 1 (3 lanes):** G1 (BurnBackend seam on burn-ndarray + tolerance harness), W1 (`SwarmApi`
  wire + node service + `swarm.db`), B1 (R2 store + presign client + egress schemes).
  - **Merge 1:** WireVersion 39→40 single coordinated commit + `just update-codec` + `codec-drift`
    green (WIRE-3). Freeze SwarmApi wire, `OpBackend` burn extension points, `PresignClient`,
    tolerance harness. Full gates + `--features burn-ndarray` + `--features iroh` (compile-only).
- **Wave 2 (3 lanes):** G2 (burn-wgpu + VRAM autotune on Vulkan/RADV), M1 (160M preset + TinyStories
  data path + safetensors), B2 (iroh gossip control plane + self-hosted relay).
  - **Merge 2:** 160M trains a step on wgpu; NET-6 + gossip conformance green; TinyStories fixture
    verified; safetensors round-trip; `tabi@1` growth (if any) synced across all three sync points.
    Freeze `IrohGossip` config surface, manifest/tokenizer conventions, 160M preset schema.
- **Wave 3 (3 lanes):** M2 (reference parity + throughput — the P1 numeric gate), B3 (live worker
  transport + e2e on real planes), BC (daemon-cloud `apps/swarm` worker: presign FIRST, then DO +
  gateway).
  - **Merge 3 = program gate.** **P1 exit** (spec §17): 160M pretrains on 1 Vulkan GPU through the
    module path; loss within tolerance of llama-burn; tokens/s within 25% or explained. **`tabi@1`
    freezes.** **Transport exit:** flagship e2e + 5 churn drills over real iroh gossip (self-hosted
    relay) + R2 (wrangler-dev/miniflare); PROTO-20 replay from the live-transport log;
    NET-1/2/3/6/8 + RUN-1/2/5/8 green; worker-connects-to-coordinator loop over WS+gossip.

## TDD test-matrix (ID → lane)

| TDD ID | Lane | | TDD ID | Lane |
|---|---|---|---|---|
| WIRE-1/2 | W1 | | NET-6 | B2 |
| WIRE-3 | Merge 1 | | NET-4 (partial dyn) | B1/B3 |
| HOST-9, HOST-3 harness | G1 | | RUN-1/2(net)/4 | B1 |
| HOST-3/8/10 (GPU) | G2 | | RUN-3/HOST-11 | M1 |
| HOST-11, safetensors | M1 | | RUN-5/8 (live), PROTO-20 (live) | B3 |
| P1 loss/throughput | M2 | | COORD-1/2/3 | BC |
| NET-1/2/3/8 | B1 | | Scheduler DIRECT ports | B1 |
| WIRE-4 | deferred (app program) | | NET-5/7 (blobs) | deferred (P4) |

## Risks (watch items for the merge owner)

1. **burn AD ≠ CPU tape bit-wise** — native lane relaxes to tolerance classes; det lane keeps
   bit-exactness. The cross-backend det-digest test is the guard (see Determinism story).
2. **Det-lane residency on GPU** — host↔device copies at the ingest boundary; a mistake desyncs
   peers (caught late by digests). G1's cross-backend digest test is the early tripwire.
3. **Meta pass at 160M is a real execute pass** (E2 deferral) — assess cost may be minutes;
   acceptable for P1, note the shape-only interpreter as follow-on.
4. **iroh pin skew** (0.97 core; NO iroh-blobs this program — it is P4) — integration owner owns the
   pin; upgrades are scheduled tasks gated on the conformance suite.
5. **Presign correctness against real R2 vs miniflare** — SigV4 quirks; the wrangler-dev smoke is
   mandatory in BC and a real-bucket checklist item for the P2 WAN gate (out of this program).
6. **Two-repo coordination** (daemon-node trunk + daemon-api branch) — no gitlink couples them; the
   only runtime contract is the presign/WS HTTP surface; freeze it early via shared DTO fixtures (BC
   exports JSON fixtures; B1/B3 tests consume them).

## Deferred (recorded, not in scope)

iroh-blobs payload plane + proto `Locator::BlobTicket` (P4, NET-5/7); RunCoordinator DO production
deployment + real R2 bucket (P2 WAN gate ceremony); CUDA/ROCm lanes; app GUI+TUI view-model
(WIRE-4); shape-only meta interpreter; hivemind-style weighted multi-corpus mixtures.

---

## Reference pack — Psyche iroh (verified anchors)

Workspace `/home/j/experiments/decentralised-llm-training/psyche`. When porting/adapting these
patterns, cite `file:line` in code comments AND in your lane ledger, and record deltas from upstream
(the TDD §6 delta-table style). These anchors are verified — trust them over the TDD's older line
hints where they conflict.

- **Endpoint build** (relay mode, discovery, QUIC transport, allowlist hooks): `shared/network/src/lib.rs:343-378`; N0 online-wait `lib.rs:381-394`.
- **Gossip init**: `Gossip::builder().max_message_size(4096)` + Hyparview (`active_view_capacity: 8`) + Plumtree (`message_id_retention: 2*60s`) `lib.rs:459-474`.
- **Topic derivation**: sha256("psyche gossip" ++ run_id) → `TopicId`, `shared/network/src/util.rs:5-13`. **Ours: blake3(envelope hash)** — record the delta.
- **Subscribe + bootstrap**: `gossip.subscribe(gossip_topic(run_id), bootstrap_endpoint_ids)` `lib.rs:337,500-503`.
- **Router/ALPNs** (gossip + blobs + model-sharing): `shared/network/src/router.rs:26-46` (NOTE: TDD's `router.rs:70,105` are its *tests*).
- **Relay selection**: `RelayKind` `lib.rs:133-157`; custom relay map `lib.rs:105-106,984-1008`; CLI default `shared/client/src/cli.rs:66-72`. No self-hosted relay deploy config exists in Psyche — **we pin relay URLs in the envelope** instead.
- **Signed gossip**: `SignedMessage::{sign_and_encode,verify_and_decode}` `shared/network/src/signed_message.rs:17-38` (postcard; **ours is canonical CBOR** via `daemon_swarm_proto::SignedMessage` — delta). Broadcast `lib.rs:568-581`; receive-verify `lib.rs:898-922`.
- **App-layer dedupe + deliberate rebroadcast** (gossip is ~99.9% delivery): dedupe `shared/client/src/state/steps.rs:427-432`; rebroadcast every 10s with nonce bump `shared/client/src/client.rs:490-505` — adopt the rebroadcast pattern for Commitments/Attestations.
- **Bootstrap-from-coordinator** (NOT init-time bootstrap): training clients init with `vec![]` peers (`architectures/decentralized/solana-client/src/app.rs:106-117`) and form the mesh from the coordinator's client list via `ensure_gossip_connected` (cap 3 neighbors) `shared/client/src/client.rs:193-195,736-799` — adopt: our roster comes from admission/Join flow.
- **Download scheduler** (per-type retry: DistroResult expo-backoff max 3; capacity gate; FIFO): actor `shared/network/src/download/scheduler.rs:208-375`; its DIRECT-portable tests `scheduler.rs:411-675`.
- **Blob tickets** (P4 seam only this program): `lib.rs:635-658` create, `lib.rs:584-632` fetch, `shared/network/src/p2p_model_sharing.rs:516-526,728-766`.
- **Crate pins**: `psyche/Cargo.toml:73-76,116-118` — iroh/iroh-relay/iroh-gossip **0.97.0**, iroh-blobs **0.99** (P4, NOT this program), plus pinned `digest = "=0.11.0-rc.10"`, `crypto-common = "0.2"`. See the "Resolved pins" section below for what our tree actually resolved.

## Reference pack — daemon-cloud

Workspace `/home/j/experiments/daemon-cloud/daemon-api`.

- Apps live in `apps/*` (pnpm workspace, `pnpm-workspace.yaml:1-3`); the **hosting split** is the model for a new domain worker: no public route, service-bound from gateway (`apps/hosting/wrangler.jsonc:6-50`, gateway binding `apps/gateway/wrangler.jsonc:6-77`).
- Gateway proxy pattern (auth + scope + forward with identity headers): `apps/gateway/src/routes/nodes.ts:19-52`; internal auth middleware `apps/hosting/src/middleware/internalAuth.ts:9-17`.
- DO patterns: `HostedNodeDO` (alarms, saga driver) `apps/hosting/src/do/hostedNodeDO.ts:32-39,133-155`; DO client stub seam `apps/hosting/src/do/client.ts:31`; sqlite-class migrations in each `wrangler.jsonc`. **No WS-hibernation or wasm usage exists yet** — both are net-new (types available, `apps/gateway/worker-configuration.d.ts:514,3293`).
- R2: binding precedent on usage-consumer (`apps/usage-consumer/wrangler.jsonc:15-19`, `src/index.ts:25-33`); **no presign code exists anywhere** — `apps/swarm` ships the first (via `aws4fetch` or R2 SDK presign).
- Tests: vitest 4 node-env with fake bindings (`apps/gateway/src/routes/nodes.test.ts:26-69`), Memory-store fakes for DO logic (`apps/hosting/src/sagas/testkit.ts:48-61,451-535`); dev via `wrangler dev` (flake ships wrangler, `flake.nix:32-57`).
- daemon-node → cloud calls go through `daemon-egress` with Bearer keys (`bins/daemon/src/main.rs:324-340`); swarm scopes (`swarm:join`) are spec-only today (`packages/shared/src/core/apiKey.ts:24-35`).

---

## Wave-0 scaffold record

Landed on `integrations/swarm-p1` (base `e2e08c3`). Commit list (oldest → newest):

| Commit | Subject |
|---|---|
| `mirror(P1-prog): ledger` | this ledger (base sha, lanes, frozen rules, determinism story, reference packs) |
| `build(deps): burn wgpu/ndarray backend lanes as opt-in daemon-train features` | burn feature plumbing |
| `build(deps): iroh transport stack behind daemon-swarm-net `iroh` feature` | iroh pins + feature gate |
| `build(nix): swarm transport + wgpu GPU training lanes` | iroh-relay tool + wasm-capable vulkan shell |
| `refactor(train): split daemon-train-worker into main/backend/transport modules` | worker module split |
| `feat(egress): EgressClient::put for presigned R2/S3 uploads` | additive egress method |

### Resolved dependency pins

- **burn** (root `[workspace.dependencies]`): unchanged requirement `burn = { version = "0.21",
  default-features = false, features = ["std", "ndarray", "autodiff"] }` (resolves burn 0.21.0). GPU
  is opt-in via `daemon-train`'s own cargo features (`crates/coprocessor/daemon-train/Cargo.toml`):
  - `cpu` (default) — current CPU/det-lane behavior, no extra deps.
  - `burn-ndarray = ["burn/ndarray"]` — the G1 native lane (ndarray+autodiff; ndarray is already on
    via the root dep, so this is a no-op-safe CI alias).
  - `wgpu = ["burn/wgpu"]` — the G2 GPU lane: burn 0.21 `wgpu` feature → burn-wgpu 0.21.0 + cubecl
    0.10.0 + wgpu 29.0.4, Vulkan/RADV at runtime. Verified `cargo check -p daemon-train`,
    `--features burn-ndarray`, `--features wgpu` all compile (wgpu needs **no** extra build-time
    system deps; runtime needs `libvulkan` — present on the default + `.#vulkan` devShell
    `LD_LIBRARY_PATH`). burn's `vulkan`/`metal`/`webgpu` features layer on top of `wgpu` if G2 wants a
    backend-locked build.
- **iroh** (root `[workspace.dependencies]`): `iroh = "1"` (1.0.2), `iroh-gossip = "0.101"`
  (0.101.0), `iroh-relay = "1"` (1.0.2). Gated behind `daemon-swarm-net`'s `iroh` feature
  (`dep:iroh`, `dep:iroh-gossip`, `dep:iroh-relay`). **NO iroh-blobs** (P4). Verified
  `cargo check -p daemon-swarm-net` (no iroh) and `--features iroh` both pass; `cargo tree` shows
  zero iroh crates on the default graph.
  - **PIN DEVIATION from the plan's "iroh 0.97" (integration owner's call, Risk 4):** iroh-base
    0.97/0.98 pull a pre-release crypto stack pinned to `sha2 =0.11.0-rc.{2,5}`, which is in the same
    0.11 semver-compat range as — and disjoint from — the **stable** `sha2 0.11.0` that
    `slack-morphism 2.22` (daemon-slack) already locks (`^0.11`). No single sha2 0.11.x satisfies
    both requirements, so **iroh 0.97/0.98 are unresolvable against the existing frozen tree**. iroh
    **1.0 dropped the sha2 dependency entirely**, resolving cleanly. iroh-gossip has no 1.0 tag; its
    0.101.0 targets `iroh ^1` (the matching release). Practical impact for **B2**: port the Psyche
    0.97 gossip patterns (reference pack) to the iroh **1.0** API and record the deltas — the
    endpoint / `Gossip::builder` / `Router` / relay shapes are largely stable across 0.97→1.0, but a
    few module paths / signatures moved. iroh-base 1.0 uses `ed25519-dalek =3.0.0-rc.0` /
    `curve25519-dalek =5.0.0-rc.0` (distinct major ranges from our stable `ed25519-dalek 2`, so they
    coexist; `--features iroh` also pulls a second `reqwest 0.13` — a duplicate-version *warning*
    only, allowed by `bans.multiple-versions = "warn"`).

### deny.toml — no change needed

`cargo deny check` is **fully green** with both new trees in `Cargo.lock` (advisories ok, bans ok,
licenses ok, sources ok). The iroh 1.0 + burn-wgpu/cubecl/wgpu trees introduced **no** new advisory,
license, or source findings — their crates are MIT/Apache/BSD (already allow-listed) and carry no
unmaintained-status advisory that is not already ignored. The `bans.multiple-versions = "warn"`
duplicates (rand, reqwest 0.12/0.13, tungstenite, …) are warnings, not gate failures. No documented
ignore was added (none warranted).

### flake.nix changes

- **Default devShell**: `iroh-relay` binary added to `packages` for B2's self-hosted relay (spec
  §7.4), pulled from the pinned nixpkgs when present (the `logos-co/nixpkgs/mingw-integration` fork
  ships **iroh-relay 1.0.0** — verified on PATH) and skipped gracefully via
  `lib.optionals (pkgs ? iroh-relay)` otherwise. **Fallback if a future nixpkgs bump drops it:**
  `cargo install --locked iroh-relay@1` into `.dev/` (a runtime tool only; the relay speaks the
  cross-1.0.x relay protocol, so the 1.0.0 tool ↔ 1.0.2 lib pin is fine).
- **`.#vulkan` devShell**: switched `craneLib` → `craneLibDev` so the wasm32-unknown-unknown rust-std
  is on the toolchain — this is now the **burn-wgpu GPU training test lane** (G2): the existing
  `vulkan-loader` on `LD_LIBRARY_PATH` resolves the RADV ICD for
  `cargo test -p daemon-train --features wgpu`, and the daemon-train guest-lifecycle tests can build
  the wasm guests (the host-only toolchain could not). `vulkan-headers`/`vulkan-loader`/`shaderc`
  were already present.
- No new package **output** for a `daemon-train-vulkan` compile lane this wave (G2 adds it in Wave 2,
  mirroring the `daemon-infer-vulkan` pattern ~`flake.nix:418`) — kept out of Wave-0 scope per the
  "don't sink hours into nix packaging" guidance; the devShell path above already gives G2 a runnable
  wgpu test lane.

### Worker binary module layout (`daemon-train-worker`)

`crates/coprocessor/daemon-train/src/bin/daemon-train-worker/` (bin `path` → `main.rs`):

- `main.rs` — `#[tokio::main]` command-dispatch loop; crate-level `#![allow(clippy::disallowed_methods)]`
  + `#![forbid(unsafe_code)]` (inherited by the submodules); shared `send`/`worker_error` helpers +
  the `SEQS`/`SEQ` micro-batch shape.
- `backend.rs` — **G2 owns.** `Probe` (`hardware`/`host_capabilities`/`host_ops`), the `AssessRun`
  envelope→`(config, module)` resolution (`ResolvedRun`, `resolve_run`, `resolve_module`,
  `module_from_env`), and the meta-mode `assess`. G2 grows real GPU `Hardware` numbers + VRAM
  autotune here.
- `transport.rs` — **B3 owns.** The `JoinRun` handler `join_and_run_round` (today the self-driven
  round loop; B3 replaces it with a live `JoinRun.coordinator` attach over `IrohGossip` + `R2Store`).

Pure mechanical split — behavior identical; all daemon-train tests green incl. the 4 `worker_protocol`
integration tests.

### `EgressClient::put` signature (for B1)

```rust
// crates/engine/daemon-egress/src/lib.rs
impl EgressRequest {
    pub fn put(url: impl Into<String>, body: Vec<u8>) -> Self;   // raw-body PUT, no forced Content-Type
}
impl EgressClient {
    pub async fn put(&self, url: &str, body: Vec<u8>, redirects: Redirects)
        -> Result<reqwest::Response, EgressError>;
}
```

- **No `Content-Type` is forced** — a presigned URL only signs the headers it was minted with, so
  forcing an unsigned `Content-Type` would break SigV4. For a content-type-signed presign, build via
  `EgressRequest::put(url, body).header("content-type", ct)` and call `EgressClient::execute`.
- **`redirects` is surfaced per-call** (house style, like `get`): presigned uploads normally pass
  `Redirects::None`. SSRF posture matches `get`: the initial URL is not re-checked (caller
  pre-flight), redirect hops are re-validated with `check_url` (a `307`/`308` into private/metadata
  space is rejected mid-chain — tested).

### Wave-1 lanes — what to know beyond the ledger

- **G1** (`--features burn-ndarray`): the lane is enabled and compiles today; `burn/ndarray` is
  already on the root dep so the feature is effectively a CI alias. Build `BurnBackend` behind the
  frozen `OpBackend` seam; the cross-backend det-digest equality test (CpuBackend vs
  BurnBackend(ndarray)) is the determinism tripwire (see Determinism story). The worker `backend`
  module is pre-split for you (G2 extends it in Wave 2; G1 stays in `daemon-train` src).
- **W1**: no scaffold blockers — `SwarmApi` + `swarm.db` + config embed are yours; do NOT bump
  WireVersion in-lane (Merge 1 does 39→40 once).
- **B1**: `EgressClient::put` is ready (signature above). The `daemon-swarm-net` `iroh` feature is
  off by default, so your `r2_store.rs`/`presign.rs`/`artifact.rs`/`fetch.rs` build on the default
  (no-iroh) gate; wire outbound HTTP through `daemon_egress::EgressClient` (raw `reqwest` is
  clippy-banned). Port the Psyche download-scheduler DIRECT tests onto the retry layer.
- **All lanes**: the frozen files (root `Cargo.toml`, `deny.toml`, `flake.nix`) are locked — route
  any new third-party dep or feature-of-a-workspace-dep-that-needs-a-root-change through the
  integration owner. Adding a *feature* of an already-declared workspace dep from your own crate's
  `Cargo.toml` (e.g. `iroh-relay/server`) is a lane-owned edit.
