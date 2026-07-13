# Swarm P1 + Transport — lane ledger **B1** (R2 store + presign client + egress schemes)

Lane **B1** of the "Swarm P1 + Transport" program, Wave 1. Owns
`crates/swarm/daemon-swarm-net/src/{r2_store.rs,presign.rs,artifact.rs,fetch.rs}` + `lib.rs`
re-exports + the crate `Cargo.toml` (dev-deps). Read `swarm-p1-ledger.md` (program ledger) and
`swarm-mvp-ledger.md` (frozen MVP surfaces) first — this ledger only records B1's deltas.

## Base + branch

- **Repo:** `daemon-node` (worktree `/home/j/experiments/daemon-worktree/swarm-runtime`).
- **Branch:** `swarm/b1`, based at `d71839a` (`mirror(P1-prog): Wave-0 scaffold record`) on
  `integrations/swarm-p1`.
- **Wave-0 facts consumed:** `EgressClient::put(url, body, redirects)` (raw-body PUT, no forced
  Content-Type — SigV4 parity); raw `reqwest::Client` is clippy-banned outside `daemon-egress`, so
  **all** HTTP goes through `EgressClient`; the `iroh` feature is B2's lane (untouched).

## Scope this wave

The `r2` payload plane + presign seam + the egress artifact schemes + the retry/scheduler layer,
all against a mock presign/object server (no real network). Frozen MVP surfaces built on:
`PayloadStore` (put/get/head), `PayloadKey`/`PayloadStat`/`ContentHash` (proto blake3), the generic
`ReceiptProducer<S: PayloadStore>` (must work unchanged over `R2Store`), the `ArtifactResolver`
scheme-dispatch + blake3 verify, and `fetch_with_fallback` + `RetryPolicy`.

## Exported seams (freeze at Merge 1)

### 1. `PresignClient` trait + the JSON DTO contract (BC implements it, Wave 3 — Risk 6)

The HTTP contract for `POST <coordinator_base>/runs/:id/presign` (spec §11.1). Frozen via the JSON
fixtures under `crates/swarm/daemon-swarm-net/tests/fixtures/presign-*.json` — BC's `apps/swarm`
worker and B3's live client both consume these bytes verbatim.

```rust
#[async_trait]
pub trait PresignClient: Send + Sync {
    async fn presign(&self, run: &RunId, req: &PresignRequest)
        -> Result<PresignResponse, SwarmNetError>;
}

// request body (JSON) — POST /api/v1/swarm/runs/:id/presign
pub struct PresignRequest {
    pub kind: ObjectKind,          // "payload" | "record-set" | "checkpoint" | "artifact"
    pub op: PresignOp,             // "put" | "get"
    pub round: Option<u64>,        // set for payload/record-set/checkpoint
    pub peer: Option<String>,      // peer node-id hex — set for payload only
    pub path: Option<String>,      // run-relative object key — set for artifact only
}
pub enum ObjectKind { Payload, RecordSet, Checkpoint, Artifact }  // serde kebab-case
pub enum PresignOp  { Put, Get }                                  // serde lowercase

// response body (JSON)
pub struct PresignResponse {
    pub url: String,                        // the presigned URL to PUT/GET
    pub expires_at: u64,                    // unix seconds; the URL cache honours this
    pub headers: BTreeMap<String, String>,  // signed headers the caller must replay (optional)
}
```

**Contract notes for BC** (the authoritative object-key layout is spec §11.3, mirrored by
`r2_store::r2_object_key`):

- `payload`     → `runs/<run>/rounds/<round>/<peer_hex>.upd`     (needs `round` + `peer`)
- `record-set`  → `runs/<run>/rounds/<round>/record-set.cbor`    (needs `round`)
- `checkpoint`  → `runs/<run>/checkpoints/round-<round>.safetensors` (needs `round`)
- `artifact`    → `runs/<run>/<path>`                             (needs `path`)
- `op=put` → presigned PUT (no forced Content-Type; if the presign signs one, return it in
  `headers` and the client replays it verbatim). `op=get` → presigned GET.
- **Generalisation delta from the brief:** the brief specified `{round, peer, kind, op}`. B1
  makes `round`/`peer` optional and adds `kind=artifact` + `path` so the **one** `/presign`
  endpoint serves both round objects (§11.1) *and* `r2://` envelope artifacts (§8), instead of a
  second endpoint. `#[serde(skip_serializing_if = "Option::is_none")]` keeps each request minimal.
  BC validates field presence per kind.

### 2. `R2Store` construction surface

```rust
pub struct R2Store<P: PresignClient> { /* presign + egress + run */ }
impl<P: PresignClient> R2Store<P> {
    pub fn new(presign: P, egress: EgressClient, run: RunId) -> Self;
    pub fn run(&self) -> &RunId;
}
impl<P: PresignClient> PayloadStore for R2Store<P> { put / get / head }
// The single source of truth for the §11.3 object-key layout (BC mints presigned URLs at the same
// keys). Takes the run + the presign request (all four kinds), erroring if a required field is
// absent — cleaner than the brief's positional `(run, round, peer, kind)` and covers `artifact`.
pub fn r2_object_key(run: &RunId, req: &PresignRequest) -> Result<String, SwarmNetError>;
```

- `put` → presign `payload/put` → `EgressClient::put(url, bytes, Redirects::None)` → 2xx → return
  `blake3(bytes)`.
- `get` → presign `payload/get` → `EgressClient::get(url, FollowValidated)` → 200 → blake3-verify
  against `expected` (reuses the frozen mismatch path) → bytes. **404/403 → `PayloadMiss`** (the
  typed miss the §6.4 stall ladder consumes; matches the `FsPayloadStore` taxonomy).
- `head` → **presigned GET + blake3 over the body** (decision below), returns `PayloadStat`.
- `ReceiptProducer<R2Store<_>>` compiles + works unchanged (NET-1 `head_emits_signed_receipt`).

### 3. Scheduler + dyn fallback (`fetch.rs`)

```rust
pub struct DownloadScheduler { /* actor handle */ }
pub struct RetryConfig { pub backoff_base: Duration, pub max_payload_retries: usize }
pub enum RetryQueueResult { Queued, MaxRetriesExceeded }
pub struct ReadyRetry { pub hash: ContentHash, pub key: PayloadKey, pub retries: usize }
impl DownloadScheduler {
    pub fn new(max_concurrent: usize, retry: RetryConfig) -> Self;
    pub async fn wait_for_capacity(&self) -> Result<(), SwarmNetError>;
    pub fn release_capacity(&self);
    pub async fn queue_failed(&self, key: PayloadKey, hash: ContentHash) -> RetryQueueResult;
    pub async fn due_retries(&self) -> Vec<ReadyRetry>;
    pub async fn remove_retry(&self, hash: ContentHash) -> bool;
}

pub async fn fetch_with_fallback_dyn(
    stores: &[&dyn PayloadStore], key: &PayloadKey,
    expected: &ContentHash, policy: RetryPolicy,
) -> Result<Vec<u8>, SwarmNetError>;
```

### 4. `fetch_record_set` helper (net side; B3 wires engine-side Wave 3)

```rust
pub async fn fetch_record_set<P: PayloadStore>(
    store: &P, key: &PayloadKey, expected: &ContentHash,
) -> Result<daemon_swarm_proto::RecordSet, SwarmNetError>;
```

Fetches (hash-verified) via `PayloadStore::get`, decodes `RecordSet`, and re-verifies
`record_set.content_hash() == expected` (locator-hash = content-address). Root verification stays
engine-side (B3). Store-agnostic on purpose: RUN-2's net half tests it over `FsPayloadStore`.

## Decisions (documented per the brief)

- **HEAD = presigned GET + hash the body** (not an HTTP `HEAD`). Rationale: `PayloadStore::head`
  takes no expected hash, and an R2 `HEAD` yields only size + an *etag/md5* — never the blake3
  `PayloadStat.hash` the `ReceiptProducer` needs. So `R2Store::head` re-fetches and hashes, exactly
  mirroring `FsPayloadStore::head`'s "re-read to attest the content hash" (store.rs). The presign
  contract therefore stays `put|get` (no `head` op). A production coordinator (BC) `HEAD`s
  server-side at zero egress and already knows the hash from the `Commitment` — that path never
  uses `R2Store::head`; this is the local/test client path.
- **hf resolver = minimal, over `EgressClient`** (NOT `daemon-models`/`hf-hub`). Rationale: (1)
  `hf-hub` does its own HTTP inside the crate, which would bypass the SSRF-safe `EgressClient` (the
  workspace invariant — all HTTP through egress); (2) `daemon-models` drags the model-management
  tree. B1 maps `hf://<repo>@<rev>/<path>` → `https://huggingface.co/<repo>/resolve/<rev>/<path>`
  and GETs it through `EgressClient` (`Redirects::FollowValidated` — HF `resolve` 302-redirects to
  the CDN), then blake3-verifies against the envelope hash. Unpinned refs (no `@<rev>`) are rejected
  with a typed `SwarmNetError::UnpinnedRevision` (spec §8). Pattern mirrored: the revision-pin shape
  of `daemon-models` `crates/providers/daemon-models/src/acquire.rs:395`
  (`Repo::with_revision(repo, Model, revision)` → the same `/resolve/<rev>/` URL space).
- **RUN-4 artifact LRU cache lands in `daemon-swarm-net`**, beside the resolver in `artifact.rs`
  (`ArtifactCache`), NOT in `daemon-swarm-run/src/data.rs`. Rationale: it caches *resolved artifact
  bytes* (the resolver's output), so it belongs next to the resolver in B1's owned file set; it
  avoids editing `data.rs` (M1→M2's file) and the resulting Wave-2 collision. B3/M2 wire it around
  the resolver at call sites. Bounded by bytes (`ArtifactCache::from_gb(data_cache_gb)` /
  `new(max_bytes)`); LRU by last-access.
- **Mock server = `wiremock`** (already a `[workspace.dependencies]` entry used by daemon-matrix /
  daemon-egress / daemon-models — adds **no** new third-party dep, needs no frozen-file change).
  A small `tests/common` harness (`MockR2`) mounts the `/presign` endpoint + a stateful `/obj/*`
  PUT/GET object store via a custom `wiremock::Respond`, and can mint expired presigns / drop
  objects (retention) for the NET-1/8 negative cases.

## Psyche download-scheduler port (code-grounded)

`fetch::DownloadScheduler` ports `psyche/shared/network/src/download/scheduler.rs:189-375` (actor)
with its DIRECT tests from `scheduler.rs:411-675`. Deltas from upstream:

- Upstream keys retry entries by `iroh_blobs::Hash` (a blob ticket); B1 keys by the blake3
  `ContentHash` and carries the `PayloadKey` (no iroh-blobs this program — P4). One retry class
  (payload, = Psyche's `DistroResult`: expo backoff `backoff_base * 2^prev_retries`,
  `max_payload_retries` default 3, time-gated `due_retries`) — Psyche's `ModelSharing`
  Parameter/Config classes are P2P-model-sharing (P4), dropped.
- Capacity gate + FIFO waiters (`VecDeque<oneshot::Sender<()>>`) + `release`-transfers-slot ported
  1:1. Ported tests: `test_capacity_grants_up_to_max`, `test_release_unblocks_waiter`,
  `test_waiters_are_served_fifo`, `test_distro_max_retries_exceeded`,
  `test_distro_retry_not_immediately_due`, `test_distro_retries_returned_and_removed`,
  `test_remove_retry`, `test_wait_for_capacity_errors_on_actor_shutdown` (renamed to the payload
  domain). `fetch_with_fallback_dyn` covers the NET-4 dyn-fallback gap.

## Additive `SwarmNetError` variants (lib.rs — I own the re-exports)

`PresignExpired(String)` (a minted URL already past `expires_at` — NET-1
`store_presign_expired_rejected`) and `UnpinnedRevision(String)` (an `hf://` ref with no pinned
revision — NET-3 `unpinned_hf_rejected`). Both additive to the `#[non_exhaustive]` enum.

## Planned slices (each `feat(swarm-net): … (green)`; lane gates green per commit)

1. `mirror(B1): ledger` (this file).
2. presign seam + DTOs + JSON fixtures + URL cache + mock harness (NET-1 presign half).
3. `R2Store: PayloadStore` + object keys §11.3 + typed miss; `ReceiptProducer<R2Store>`
   (NET-1 full, NET-8).
4. egress artifact schemes (`https`/`r2`/`hf`) + `ArtifactCache` (NET-2/3, RUN-4).
5. `DownloadScheduler` port + `fetch_with_fallback_dyn` + `fetch_record_set` (scheduler DIRECT
   ports, NET-4 dyn, RUN-2 net half).

## What Merge 1 / B2 / B3 / BC must know

- **Merge 1:** freeze the `PresignClient` trait + the `presign-*.json` fixtures as the node↔cloud
  HTTP contract. No new third-party deps, no frozen-file edits, no WireVersion touch.
- **BC (Wave 3):** implement `/presign` to the fixtures verbatim; object keys per §11.3 /
  `r2_object_key`; `op=put` must not require a Content-Type unless it returns it in `headers`.
- **B3 (Wave 3):** `fetch_record_set` is store-generic — wire it into `engine.rs::verify_record_set`
  with the run's `R2Store` (record-set via `ObjectKind::RecordSet` presign) or the inline set at
  small rosters. `fetch_with_fallback_dyn` + `DownloadScheduler` are the retry surface for the
  concurrent in-peer fetch task. Swap wiremock → wrangler-dev when BC's endpoint lands.
- **B2:** untouched — the `iroh` feature and `iroh_gossip.rs` are yours; B1 builds on the default
  (no-iroh) gate.

## Landed (final)

Commits on `swarm/b1` (base `d71839a`):

1. `mirror(B1): ledger` — this file.
2. `feat(swarm-net): r2 store + presign seam + egress schemes + scheduler (green)` — the full
   implementation + in-crate tests + `tests/fixtures/presign-*.json` + the `wiremock` MockR2 harness
   + `Cargo.lock` dep edges.
3. `mirror(B1): landed record` — this section.

**Test count:** `cargo test -p daemon-swarm-net` = **67 green** (was 40 pre-lane): NET-1
(`store_presign_roundtrip`, `store_presign_expired_rejected`, `head_emits_signed_receipt`), NET-2
(`verify_artifact_ok`/`verify_artifact_tamper` + blake3 golden), NET-3 (`resolve_hf_pinned_ok`,
`unpinned_hf_rejected`, `r2_to_presign`, + `https_scheme_routes_through_egress`), NET-8
(`retained_object_fetchable`, `expired_object_typed_miss`), NET-4 (`dyn_fallback_r2_miss_to_fs` +
two Fs dyn cases), RUN-2 net (`tampered_set_object_rejected`, `record_set_round_trips…`), RUN-4
(`artifact_cache_lru_evicts` + oversize/from_gb), the 8 ported Psyche scheduler tests, the presign
fixture-contract + URL-cache tests.

**Gates (all green):** `cargo fmt --check`; `cargo clippy --workspace --all-targets -D warnings`;
`cargo test --workspace` (only the two documented pre-existing `daemon-conformance` flakes —
`node::history::…` + `node::detached_delegation::…` — fail under the parallel run and **pass in
isolation**, never touched by this lane); `cargo test -p daemon-swarm-net`; `typos docs/specs`;
plus the Merge-1 compile-only `cargo check -p daemon-swarm-net --features iroh`.

**Deviations from the brief:**
- `PresignRequest` generalised to `{kind, op, round?, peer?, path?}` with `kind=artifact` (one
  `/presign` endpoint for both round objects and `r2://` artifacts). Documented above; BC validates
  per kind.
- `R2Store::head` = presigned GET + hash the body (no `head` op in the contract). Documented above.
- hf resolver hand-rolled over `EgressClient` (not `daemon-models`/`hf-hub`), keeping all HTTP on the
  SSRF-safe path. `ArtifactResolver::with_hf_endpoint` added for a private mirror / hermetic tests
  (mirrors `HfClient::with_endpoint`).
- RUN-4 `ArtifactCache` landed in `daemon-swarm-net` (`artifact.rs`), NOT `daemon-swarm-run/data.rs`
  — `data.rs` was left untouched (no Wave-2 collision with M1).
- `r2_object_key(run, &PresignRequest)` instead of the brief's positional signature (covers all four
  object kinds; validates required fields).
