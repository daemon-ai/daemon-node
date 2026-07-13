# Swarm P2 WAN — lane ledger **A1** (node↔cloud control plane)

Lane **A1** of the "Swarm P2 WAN Program", Wave 1 (the first blocking link in the integration
chain). Owns the node WS coordinator client, the dual-plane (WS + gossip) control surface, the
run-discovery + AssessRun join flow, and the v42 wire delta. Read the program ledger
(`docs/specs/swarm-p2-ledger.md`) + P1 ledger (`docs/specs/swarm-p1-ledger.md`) first — this ledger
records only A1's deltas.

- **Repo / branch:** `daemon-node`, `swarm/a1`, base `3cd43c1` (Wave-0 trunk).
- **Worktree:** `/home/j/experiments/daemon-worktree/p2-a1`.
- **Files owned:** `crates/swarm/daemon-swarm-net/{Cargo.toml,src/ws_client.rs,src/dual_plane.rs,src/registry.rs}`
  + `src/lib.rs` re-exports; `crates/swarm/daemon-swarm-node/{Cargo.toml,src/discovery.rs,src/service.rs,src/lib.rs,tests/service.rs}`;
  `crates/contracts/daemon-api/{src/swarm.rs,src/wire.rs,daemon-api.cddl,tests/swarm_conformance.rs}`
  + `crates/contracts/daemon-common/src/lib.rs` (the single v42 bump); one boot-site line in
  `bins/daemon/src/main.rs` (the additive `SwarmServiceParts.discovery` field).
- **No frozen-file edits:** root `Cargo.toml` / `deny.toml` / `flake.nix` untouched (the WS stack
  reuses the in-tree `tokio-tungstenite 0.29` workspace dep; the rustls TLS feature is a lane-owned
  edit in `daemon-swarm-net`'s own manifest). `Cargo.lock` is the only co-touched file.

---

## 1. Node WS coordinator client (`daemon-swarm-net::ws_client`, `ws` feature)

A new module behind a `ws` cargo feature (mirrors `iroh`; off by default so the default workspace
build compiles no WS/TLS tree). It dials **out** to the RunCoordinatorDO `GET {base}/runs/:id/ws`
surface over `wss://`, speaking canonical-CBOR `SignedMessage` frames both ways, and presents as a
`ControlPlane` so the frozen `RoundEngine` runs over it unchanged.

- **WS stack:** reuses `tokio-tungstenite 0.29` (Wave-0 decision), TLS via
  `features = ["rustls-tls-webpki-roots"]` in `daemon-swarm-net`'s manifest (matches the tree's
  rustls/aws-lc posture; no native-tls; no second WS lib). `daemon-host::ws` is the in-tree usage
  pattern followed (binary message == one CBOR frame, no length prefix).
- **Framing:** one WS **binary** message == exactly one canonical-CBOR `SignedMessage` frame, no u32
  length prefix — byte-identical to what the DO's `decodeSignedFrame` consumes.
- **Delivery contract = Loopback/Iroh:** a publish **self-delivers** locally (the DO never echoes a
  peer's own frame — `broadcast([bytes], ws)` excludes the sender) and every frame is deduped by
  content hash (`Deduper`), so a WS+gossip double-arrival delivers once (NET-6).
- **Reconnect + resubscribe:** a background task reconnects with exponential backoff on any drop and
  re-sends the registered resubscribe frames (the peer's signed `Join`) on every (re)connect;
  publishes issued while disconnected buffer and flush on reconnect.
- **Auth (never hardcoded — from `JoinRun.credentials`/config):** `WsAuth::Bearer` →
  `Authorization: Bearer <token>` (the gateway `swarm:join` path); `WsAuth::Internal{org_id,actor}`
  → `x-daemon-org-id`/`x-daemon-actor` (the direct-to-`apps/swarm` dev path); `WsAuth::None` (bare).
- **`base_url` swap is trivial** (gateway ↔ wrangler-dev ↔ mock): only `WsConfig.base_url` changes.

### Exported seam (FREEZE at Merge 1) — verbatim

```rust
// daemon_swarm_net::ws_client  (feature = "ws")

pub enum WsAuth {
    None,
    Bearer(String),
    Internal { org_id: String, actor: String },
}

pub struct ReconnectConfig {
    pub enabled: bool,
    pub initial_backoff: Duration,   // default 500ms; doubles per consecutive failed attempt
    pub max_backoff: Duration,       // default 30s
    pub max_attempts: Option<u32>,   // None = retry forever
}
impl Default for ReconnectConfig { /* enabled, 500ms, 30s, None */ }

pub struct WsConfig {
    pub base_url: String,            // e.g. https://api.daemon.ai/api/v1/swarm  (or http://127.0.0.1:8795/api/v1/swarm)
    pub run_id: String,
    pub auth: WsAuth,
    pub reconnect: ReconnectConfig,
}
impl WsConfig { pub fn endpoint(&self) -> String; }   // wss://…/runs/:id/ws

pub struct WsControlPlane { /* … */ }
impl WsControlPlane {
    pub async fn connect(config: WsConfig) -> Result<Self, SwarmNetError>;  // eager first connect
    pub fn endpoint(&self) -> &str;
    pub fn add_resubscribe_frame(&self, frame: Vec<u8>);  // the peer's Join; re-sent on every (re)connect
    pub fn connect_count(&self) -> u64;    // successful (re)connections (reconnect-drill signal)
    pub fn is_connected(&self) -> bool;
    pub async fn shutdown(&self);          // abort the background task (also on Drop)
}
#[async_trait] impl ControlPlane for WsControlPlane { /* publish / subscribe */ }
```

## 2. Dual-plane control surface (`daemon-swarm-net::dual_plane::DualPlane`)

Runs the WS plane and the iroh gossip mesh (or any `Arc<dyn ControlPlane>` set) at once with
cross-plane content-hash dedupe: a `publish` fans out on **every** inner plane; a subscription merges
all inner subs behind one `Deduper`, so the same `SignedMessage` arriving on both WS and gossip is
delivered exactly once. `publish` succeeds if **any** plane accepts it, so one degraded plane never
fails the publish (spec §7.1: the coordinator WS carries the same messages if gossip degrades). It is
feature-independent (needs neither `ws` nor `iroh` itself — the caller supplies the concrete planes).

### Exported seam (FREEZE at Merge 1) — verbatim

```rust
// daemon_swarm_net::dual_plane
pub struct DualPlane { /* … */ }
impl DualPlane {
    pub fn new(planes: Vec<Arc<dyn ControlPlane>>) -> Self;
    pub fn pair(a: Arc<dyn ControlPlane>, b: Arc<dyn ControlPlane>) -> Self;  // WS + gossip
    pub fn plane_count(&self) -> usize;
}
#[async_trait] impl ControlPlane for DualPlane { /* publish→all; subscribe→merged+deduped */ }
```

**Dedupe design:** the WS plane and the iroh gossip plane each already self-deliver + dedupe on
their own `Deduper`. `DualPlane::subscribe` subscribes to each inner plane and forwards through a
**per-subscription** `Deduper` (proto blake3 over the raw frame bytes), so whichever plane arrives
first delivers and the other is dropped. Verified end-to-end: two dual-plane peers sharing a mock WS
coordinator + a loopback gossip bus — a publish from peer 1 reaches peer 2 over BOTH planes (WS relay
+ gossip fanout) and is delivered once (`dual_plane_ws::same_frame_over_ws_and_gossip_delivers_once`).

## 3. Run discovery + AssessRun join flow

- **`daemon-swarm-net::registry::RegistryClient`** (new): `GET {base}/runs` (list) + `GET
  {base}/runs/:id` (detail; `Ok(None)` on 404) + `fetch_envelope` (presigned `GET` of the run-relative
  `envelope.cbor`, §11.3, then **blake3-verify** vs the descriptor's `envelope_hash` — a mismatch is
  `SwarmNetError::HashMismatch`, the §12 tamper path). All HTTP rides `daemon_egress::EgressClient`
  (raw `reqwest` banned); auth is Bearer (gateway) or the internal identity headers (dev). The
  `RunDescriptor` DTO mirrors the cloud `apps/swarm` `RunDescriptor` (`{ "data": … }`-wrapped).
- **`daemon-swarm-node::discovery`** (new): the `RunDiscovery` seam (`list_runs`/`get_run`/
  `fetch_envelope`) + `EgressRunDiscovery` (wraps a `RegistryClient`; its base is the coordinator
  endpoint). Testable with a fake (like `WorkerControl`).
- **`SwarmService::swarm_join` rewire:** when a discovery seam is configured it resolves the run,
  fetches + verifies the frozen envelope, and runs the worker's real §6.5 `AssessRun` (`worker.assess`)
  **before** `JoinRun`, taking the coordinator from discovery; the assess verdict is mapped onto the
  node-computed `SwarmEligibility` and persisted (the app renders it, ADR-003). With **no** discovery
  configured it falls back to the W1 probe-based eligibility against the allowlisted coordinator
  (offline / no-registry path), so existing behavior is preserved. Wiring, not a wire change — the
  `swarm_run_list`/`swarm_run_detail` DTOs already carry eligibility.

### Exported seam (FREEZE at Merge 1) — verbatim

```rust
// daemon_swarm_node::discovery
pub struct DiscoveredRun { pub run_id: String, pub coordinator: String, pub envelope_hash: String, pub proto_version: u32 }
#[async_trait] pub trait RunDiscovery: Send + Sync {
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, SwarmError>;
    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, SwarmError>;
    async fn fetch_envelope(&self, run_id: &str) -> Result<Vec<u8>, SwarmError>;
}
pub struct EgressRunDiscovery { /* … */ }
impl EgressRunDiscovery { pub fn new(registry: daemon_swarm_net::RegistryClient) -> Self; }

// daemon_swarm_node::service — SwarmServiceParts gains an additive field:
pub struct SwarmServiceParts { /* config, store, worker, feed, */ pub discovery: Option<Arc<dyn RunDiscovery>> }
```

## 4. v42 wire delta (additive; targets v42 per the W1/Merge-1 precedent)

`SwarmHardwareReport.shared_mb: u64` — the app-facing mirror of the worker's unified-memory (GTT)
spillover the node already probes (`daemon_swarm_run::protocol::Hardware.shared_mb`), so the GUI shows
the true effective device budget on integrated/UMA boxes (`vram_mb + 90%·shared_mb`, §10.5). This IS
a wire change (the P1 Merge-2 recorded follow-on), landed exactly like the Merge-1 precedent:

- `daemon_api::SwarmHardwareReport` gains `#[serde(default)] pub shared_mb: u64` (after `vram_mb`).
- `daemon-api.cddl`: `swarm-hardware-report` gains `"shared_mb": uint64`; header comment `current = 42`.
- `daemon-common::WireVersion::CURRENT` bumped **41 → 42** (single coordinated bump) with the doc note.
- Pinned gate retargeted: `contract_wire_version_is_v41` → `contract_wire_version_is_v42` (asserts 42).
- Conformance `swarm_conformance.rs` `hardware()` fixture carries `shared_mb`; WIRE-1 (4) + WIRE-2
  (`--features arbitrary`, 75) green.
- Node mapping: `service::hardware_report` sets `shared_mb: hw.shared_mb`.

**Merge-1 note:** the 41→42 bump already lives on `swarm/a1` — the integration owner must NOT re-bump.
A1 is the only Wave-1 wire change (B1 + the cloud lane touch no `daemon-api` wire), so the merge is
clean. **Superproject follow-on (human, signed):** `just update-codec` + `just codec-drift` to
regenerate `daemon-app`'s vendored C codec from the v42 CDDL (grows the `swarm-hardware-report` arm) —
the daemon-node half is done here; the app codec is one wire version behind until then.

## 5. DO-contract discrepancies found (vs spec §11)

Implemented what the DO **actually speaks** (`daemon-cloud/daemon-api/apps/swarm/src/coordinator/{do,machine}.ts`,
`index.ts`, `packages/shared/src/swarm/{messages,types,keys}.ts`), not the spec's idealization. Findings:

1. **Dissemination excludes the sender.** `webSocketMessage` relays an inbound frame to the *other*
   peers (`broadcast([bytes], ws)` with `except: ws`) and broadcasts coordinator emissions to *all*.
   So the WS client must self-deliver its own publish locally (it will never receive its own frame
   back) — matched exactly (mirrors Loopback/Iroh). Spec §7.1 describes "disseminate to all"; the DO's
   no-echo-to-sender is the concrete behavior. Not blocking.
2. **HTTP `/msg` fallback exists alongside `/ws`.** `POST {base}/runs/:id/msg` and `/join` forward a
   signed frame to the DO out-of-band (no socket). A1 uses the WS path only; the `/msg` path is a
   viable degraded carrier (recorded for A3 if a peer cannot hold a socket). Not used this wave.
3. **Envelope fetch is a presigned artifact GET, not a dedicated route.** The frozen envelope lives at
   the R2 key `runs/<run>/envelope.cbor` (`envelopeKey`); there is no `GET /runs/:id/envelope`. A1
   fetches it via `presign{kind:artifact,op:get,path:"envelope.cbor"}` → object GET (matches §11.3).
4. **DO round-timeouts are coarse T0 constants** (`WARMUP_TIMEOUT=30`, `ROUND_TIMEOUT=60`,
   `COOLDOWN=5`), not envelope-driven (the DO's own recorded follow-on). Irrelevant to the A1 client
   (it renders coordinator frames; it does not drive timeouts) — recorded for the Merge-2 live loop.
5. **DO is still the Phase-1 TS shell** (no wasm `tick`): its `RoundOpen` seed schedule / assignment /
   phase arithmetic are NOT yet asserted bit-equal to the Rust `LocalCoordinator` (COORD-1
   `dual_shell_parity`, the A4 lane). The A1 client is agnostic to this (opaque signed frames both
   ways), so it is unaffected; flagged for the Merge-1/Merge-2 cross-lane parity check.

**No silent adaptation:** the framing golden (`ws-frame-commitment.cbor`) pins the exact canonical-CBOR
`SignedMessage` bytes the DO's `decodeSignedFrame` consumes; a regression fails loud. The cross-lane
live check (WS client ↔ real DO on wrangler-dev) is **Merge 1 / C1's** to run — A1 did NOT stand up
wrangler-dev (structured so a `base_url` swap is the only change).

## 6. Tests (all green; `nix develop`)

- **WS client** (`ws_control_plane.rs`, feature `ws`): **6** (+1 `#[ignore]` fixture regenerator) —
  framing golden (byte-exact + round-trip + verify), publish→relay-to-others + self-deliver,
  coordinator broadcast→all, reconnect+resubscribe drill, Bearer auth header, internal identity headers.
- **ControlPlane conformance over mock-WS** (`control_plane_conformance.rs`): **+2** (`ws_conformance_
  fanout`, `ws_conformance_dedupe`) — B2's parametric suite passes over the WS plane against the
  in-process mock DO (relay-to-others framing). (Loopback 2 + Iroh 2 unchanged.)
- **Dual-plane dedupe**: `dual_plane_ws::same_frame_over_ws_and_gossip_delivers_once` (**1**) + unit
  `dual_plane` (**2**).
- **Discovery+assess**: `discovery.rs` mock-registry integration (**3**: list+get+404, envelope
  fetch+verify, hash-mismatch reject) + node `service::join_discovers_fetches_envelope_and_assesses`
  (node suite **8** total).
- **ws_client unit** (**3**: endpoint scheme-swap, backoff monotonic+capped, auth stamping);
  **registry unit** (**1**).
- **Wire v42**: `contract_wire_version_is_v42`; `swarm_conformance` **4**; `daemon-api --features
  arbitrary` **75** protocol_conformance.

## 7. Gate results (HEAD of `swarm/a1`)

All green except the documented pre-existing `daemon-conformance` detached-delegation flake.

- `cargo fmt --all --check` ✓ · `cargo clippy --workspace --all-targets -- -D warnings` ✓.
- Feature-combo clippy `-D warnings`: `-p daemon-swarm-net --features ws` ✓ · `--features ws,iroh` ✓ ·
  `--features iroh` ✓ (default via workspace) · `-p daemon-api --features arbitrary` ✓.
- `cargo deny check` ✓ (advisories/bans/licenses/sources ok — the `ws` rustls-webpki-roots feature
  added nothing that trips the gate; tungstenite was already in the lock).
- `cargo test --workspace` ✓ **except** the known `daemon-conformance` detached-delegation trio
  (`detached_fanout_materializes_distinct_children` failed under the full parallel run; **re-verified
  5/5 green in isolation** — `cargo test -p daemon-conformance --lib node::detached_delegation`; never
  modified; A1 touches no `daemon-conformance`).
- `-p daemon-swarm-net --features ws` suites ✓ · `-p daemon-swarm-node` (8) ✓ · `-p daemon-common` ✓.
- `cargo run -p xtask -- build-guests` ✓ · both `wasm32-unknown-unknown` builds
  (`daemon-swarm-{proto,coordinator}`) ✓ · `typos docs/specs` ✓.

## 8. Deviations / notes for Merge-1 / C1 / A3

- **Guest manifest (`guests/guests.blake3`) NOT committed by A1.** A fresh `build-guests` in this
  worktree produces different guest-wasm bytes than the trunk-committed manifest, though A1 touches no
  guest source (the guest workspace does not compile `daemon-common`/`daemon-swarm-proto`; only its own
  `daemon-train-sdk`, and `guests/Cargo.lock` is unchanged) — a cross-worktree rebuild artifact, not an
  A1 change. Per "rebuild, don't override": A1 rebuilt (so the on-disk manifest matches the on-disk
  guests and the wasm suites pass) but leaves the committed manifest as the trunk's. **Integration
  owner:** rebuild guests in the merge worktree before the wasm gate (byte-reproducibility across
  worktrees is not holding — worth a Wave-1 look, but it is not an A1 regression).
- **`bins/daemon` boot site** carries `discovery: None` (probe fallback). Wiring a
  `RegistryClient`-backed `EgressRunDiscovery` at boot needs the coordinator registry base +
  `swarm:*` credentials plumbed from `[swarm]` config — a small follow-on (A3 / integration).
- **A3 (worker live attach):** construct the live plane as `DualPlane::pair(WsControlPlane, IrohGossip)`
  — the WS coordinator base + auth come from `JoinRun.coordinator` / `JoinRun.credentials`; register the
  signed `Join` via `add_resubscribe_frame` so a reconnect re-admits. The dual plane dedupes WS+gossip.
- **C1 / Merge-1 live check:** point a `WsControlPlane` at wrangler-dev
  (`base_url = http://127.0.0.1:8795/api/v1/swarm`) after `POST {base}/runs` seeds the run + DO; the
  framing golden bytes are the contract the DO must accept/emit.
