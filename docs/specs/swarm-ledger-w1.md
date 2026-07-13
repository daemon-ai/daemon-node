# Swarm P1 — lane W1 ledger (SwarmApi wire + node service)

Lane **W1** (wire/node) of the *Swarm P1 + Transport* program, Wave 1. This is the lane-local
coordination record: base sha, scope, exported seams, planned slices, and — critically — the exact
edits **Merge 1** must make (the single coordinated `WireVersion` 39→40 bump is NOT done in this
lane). Read the program ledger `swarm-p1-ledger.md` and the frozen-interface inventory
`swarm-mvp-ledger.md` first; this file only carries W1's deltas on top of them.

## Base + branch

- **Repo:** `daemon-node` (standalone submodule checkout).
- **Base commit:** `d71839a` (`mirror(P1-prog): Wave-0 scaffold record`) on `integrations/swarm-p1`.
- **Branch / worktree:** `swarm/w1` @ `/home/j/experiments/daemon-worktree/swarm-proto`.
- **Merges back into** `integrations/swarm-p1` at Merge 1.

## Scope (per the program plan "W1 — SwarmApi wire + node service")

1. `SwarmApi` sub-trait in `daemon-api` (all methods default `Err(ApiError::Unsupported)`), added to
   the `NodeApi` super-trait bound — the ModelApi precedent (`lib.rs` ~1523 / bound ~2249).
2. `ApiRequest::Swarm*` / `ApiResponse::Swarm*` wire variants mirroring the `Model*` block, `swarm-*`
   CDDL rules in `daemon-api.cddl`, conformance fixtures + negatives (WIRE-1) and `arbitrary` proptest
   coverage (WIRE-2).
3. `SwarmConfig` embed in `NodeConfig` (`bins/daemon/src/config.rs`) with `#[serde(default)]` + a
   figment layering test.
4. Node `SwarmService` (new crate `crates/swarm/daemon-swarm-node`): owns a
   `daemon_train_client::TrainSupervisor`, translates worker `Event`s → node feed events, persists
   durable state to `swarm.db` (tables per spec §10.3), and re-issues `JoinRun` for every persisted
   active intent on start (durable-intent re-convergence). OFF by default (`[swarm] enabled = false`).
5. `SwarmApi` handlers wired onto the node's `NodeApiImpl` (forwarding seam) mapping requests →
   supervisor commands + store reads.
6. `just swarm-dev` recipe — proposed as a diff below (the superproject justfile is outside this repo;
   no daemon-node justfile exists, so there is no in-repo variant to add).

## CRITICAL — the version rule (what Merge 1 must do; W1 does NOT)

`API_WIRE_VERSION` is `daemon_common::WireVersion::CURRENT` (`daemon-common/src/lib.rs:761`,
`pub const CURRENT: Self = Self(39)`). **W1 leaves it at 39.** The pinned assertions
(`daemon-api/src/wire.rs` `contract_wire_version_is_v39`, ~2360-2366) stay green in this lane. All
W1 CDDL + wire additions are structured so the bump is the *only* remaining step.

**Merge 1 does exactly this (single coordinated commit):**

1. `daemon-common/src/lib.rs:761` — `pub const CURRENT: Self = Self(39);` → `Self(40)`.
2. `daemon-api/src/wire.rs` — rename `contract_wire_version_is_v39` → `..._v40` and change both
   assertions from `WireVersion(39)` to `WireVersion(40)`.
3. `daemon-api/daemon-api.cddl` — the header comments `wire_version = uint ; ... current = 39` and
   `wire_version (daemon-common WireVersion::CURRENT)` → 40 (comment-only; no shape change).
4. `just update-codec` (regenerate the vendored C codec into `daemon-app`) + `just codec-drift`
   green (WIRE-3). The new `swarm-*` rules are additive; the generated codec grows the swarm arms.
5. Full workspace gates + `--features arbitrary` (WIRE-2) + `--features burn-ndarray`/`--features iroh`
   (compile-only, other lanes).

Nothing else in W1's surface needs a version-coupled edit — the `swarm-*` rules are appended to the
`api-request` / `api-response` / `node-event` unions additively.

## Exported seams (freeze at Merge 1)

### 1. `SwarmApi` trait (`daemon-api`, in `NodeApi` super-trait bound)

All methods default to `Err(ApiError::Unsupported)` / empty; the node binds the real impl.

```rust
#[async_trait]
pub trait SwarmApi: Send + Sync {
    async fn swarm_run_list(&self) -> Result<Vec<SwarmRunSummary>, ApiError>;
    async fn swarm_run_detail(&self, run_id: String) -> Result<Option<SwarmRunDetail>, ApiError>;
    async fn swarm_join(&self, run_id: String, policy: SwarmPolicy, op_id: String) -> Result<(), ApiError>;
    async fn swarm_leave(&self, run_id: String, mode: SwarmLeaveMode, op_id: String) -> Result<(), ApiError>;
    async fn swarm_set_policy(&self, policy: SwarmPolicy) -> Result<(), ApiError>;
    async fn swarm_hardware_report(&self) -> Result<SwarmHardwareReport, ApiError>;
    async fn swarm_subscribe(&self, run_id: Option<String>) -> Result<SwarmEventStream, ApiError>;
}
```

DTOs (all `#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]`, serde, `PartialEq`/`Eq`):
`SwarmPolicyMode` (`always`/`idle`/`scheduled`/`manual`), `SwarmPolicy`, `SwarmEligibility`,
`SwarmCapabilities`, `SwarmHardwareReport`, `SwarmContribution`, `SwarmRunSummary`, `SwarmRunDetail`,
`SwarmLeaveMode` (`graceful`/`immediate`), `SwarmEvent` (+ `SwarmEventKind`).
`SwarmEventStream = BoxStream<'static, SwarmEvent>`.

**Eligibility is node-computed** (ADR-003 mirror): `SwarmRunSummary` carries `SwarmEligibility`; the
app renders joinable-or-why-not, never re-derives it. Experiment-opaque fields stay opaque (the seam
rule): the DTOs carry `phase`/`policy`/`eligibility`/contribution counters and never any experiment
config or module bytes.

### 2. Subscription seam (rides the existing feed — no new transport)

`swarm_subscribe` returns a `BoxStream<SwarmEvent>` (the in-process / service-broadcast seam,
mirroring `events_subscribe`'s `NodeEventStream`). Over the socket mux, live swarm updates ride the
**existing** `events_subscribe` channel via a new payload-free pointer
`NodeEvent::SwarmChanged { run_id: Option<String>, rev }` (ADR-003 invalidation-pointer style — the
client refetches `SwarmRunDetail`, whose `recent_events` carries the windowed `SwarmEvent`s, ADR-007
§10.3). No new socket pump / streaming request variant is added (deliberate — "ride the existing
feed machinery").

### 3. `SwarmService` construction surface + feed event types (`daemon-swarm-node`)

- `SwarmStore::open(path) -> Result<SwarmStore>` — `swarm.db`, rusqlite + `rusqlite_migration`.
- `SwarmService::new(SwarmServiceConfig { config: SwarmConfig, store, supervisor_factory, feed })`.
- `SwarmService::start()` — no-op when `!config.enabled` (never spawns a worker); when enabled,
  re-issues `JoinRun` for every persisted **active** join-intent (re-convergence).
- `SwarmService::handle_worker_event(&Event)` — translates a worker `Event` → persists to
  `swarm_events`/`swarm_contrib` + broadcasts a `SwarmEvent` + emits `NodeEvent::SwarmChanged`.
- Implements `daemon_api::SwarmApi`.

### 4. `swarm.db` schema (spec §10.3)

```sql
swarm_runs(run_id TEXT PK, coordinator TEXT, policy_json TEXT, desired_state TEXT,
           credentials_ref TEXT, last_phase TEXT, last_step INTEGER, updated_ms INTEGER)
swarm_contrib(run_id TEXT PK, rounds INTEGER, tokens INTEGER, bytes_up INTEGER,
              bytes_down INTEGER, witness_count INTEGER, checkpoint_credits INTEGER)
swarm_events(rowid INTEGER PK AUTOINCREMENT, run_id TEXT, ts_ms INTEGER, kind TEXT, body BLOB)
```

`desired_state` = the durable join-intent flag (`joined`/`left`); restart re-convergence reads
`swarm_runs WHERE desired_state = 'joined'`. `swarm_events` is windowed (ADR-007): capped ring per
run, pruned on insert. Intents idempotent via op-id (ADR-006) at the API layer.

### CDDL rules added (`daemon-api.cddl`, additive — appended to the unions)

`swarm-policy-mode`, `swarm-policy`, `swarm-eligibility`, `swarm-capabilities`,
`swarm-hardware-report`, `swarm-contribution`, `swarm-run-summary`, `swarm-run-detail`,
`swarm-leave-mode`, `swarm-event`, `swarm-event-kind`; requests `request-swarm-run-list`,
`request-swarm-run-detail`, `request-swarm-join`, `request-swarm-leave`, `request-swarm-set-policy`,
`request-swarm-hardware-report`; responses `response-swarm-run-list`, `response-swarm-run-detail`,
`response-swarm-hardware-report`; and `node-event-swarm-changed`. Appended to `api-request`,
`api-response`, and `node-event` respectively.

## Dependency note (bins/daemon → daemon-swarm-run + daemon-swarm-node)

Embedding `SwarmConfig` makes `bins/daemon` depend on `daemon-swarm-run` (config source), and the
service wiring adds `daemon-swarm-node` + `daemon-train-client`. Verified with `cargo tree` that this
drags **no** heavy tree onto the default gate: `daemon-swarm-run` → `daemon-swarm-net` (reqwest via
`daemon-egress`, already in-tree) + `daemon-swarm-proto` (ed25519, already in-tree); **no** burn, **no**
wasmtime, **no** iroh (iroh is behind `daemon-swarm-net`'s off-by-default `iroh` feature;
`daemon-swarm-run` has no dep on `daemon-train`/`-sdk`). `daemon-train-client` links only light
node-side crates (daemon-provision/daemon-common/tokio). (cargo tree evidence appended at finalize.)

## Planned slices (TDD, green each commit)

1. `mirror(W1): ledger` (this file).
2. `feat(api): SwarmApi sub-trait + DTOs, bound into NodeApi`.
3. `feat(api): Request/Response::Swarm* wire + swarm-* CDDL + conformance + arbitrary` (WIRE-1/2).
4. `feat(node): embed SwarmConfig in NodeConfig + figment layering test`.
5. `feat(swarm-run): daemon-swarm-node SwarmStore (swarm.db) + migrations` (+ idempotence test).
6. `feat(swarm-run): SwarmService — event fanout, join-intent re-convergence, disabled-by-default`.
7. `feat(node): NodeApiImpl SwarmApi forwarding seam`.
8. `mirror(W1): finalize ledger` (evidence, test counts, cargo-tree).

## `just swarm-dev` recipe (proposed diff — superproject justfile, human applies)

The superproject `justfile` is outside this submodule. There is **no** daemon-node-local justfile, so
this is the only recipe. Model: the existing `dev-run` recipes wrap `nix develop` + env vars; this one
builds the worker + runs the local swarm driver.

```diff
--- a/justfile
+++ b/justfile
@@
+# Run a local swarm dev loop: build the daemon-train worker, then drive bins/swarm-local against it
+# (CPU backend, fs payload store, loopback transport). Off-gate deps (burn/iroh) stay untouched.
+# PEERS / PROFILE override the roster size and comm profile.
+swarm-dev PEERS="3" PROFILE="sparse_loco":
+    nix develop --command bash -c '\
+      cargo build -p daemon-train --bin daemon-train-worker && \
+      DAEMON_TRAIN_MODULE="$(pwd)/guests/target/wasm32-unknown-unknown/release/tiny_llama.wasm" \
+      cargo run -p daemon-swarm-run --bin swarm-local -- \
+        --backend worker \
+        --worker-bin "$(pwd)/target/debug/daemon-train-worker" \
+        --peers {{PEERS}} --profile {{PROFILE}} examples/local-demo.toml'
```

(Kept out of `just e2e` for now; the sealed-bundle parity path is unaffected. `bins/swarm-local`
lands/grows in lane B3 — this recipe is the wiring the integration owner applies once that bin exists;
until then it documents the intended dev loop.)

## Deviations (recorded)

- **SwarmService lives in a new crate `crates/swarm/daemon-swarm-node`**, not inside `bins/daemon`.
  Rationale: the `crates/*/*` glob picks it up with no frozen-root edit; it keeps the node service
  fully unit-testable in isolation (fake worker), and matches the house convention of a per-subsystem
  crate + separate DB (the `daemon-auth` / `auth.db` precedent).
- **`swarm_subscribe` rides the existing node event feed** (`NodeEvent::SwarmChanged` pointer) rather
  than a new streaming wire request + socket pump. Faithful to §10.4 ("over the existing mux
  subscription channel") and the brief's "no new transport". The `swarm_subscribe` trait method
  returns a `BoxStream<SwarmEvent>` for the in-process transport + the service's own broadcast.
- **`daemon-train-client` is lane B3's** (read-only for W1). Its `TrainSupervisor` exposes no event
  stream yet, so W1's `SwarmService::handle_worker_event` is the translation seam B3 wires the live
  worker event stream into during Wave 3. W1 tests drive it directly.
