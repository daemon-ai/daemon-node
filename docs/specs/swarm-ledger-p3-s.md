# Swarm P3 Follow-ons — lane ledger **S** (160M fleet staging / artifact distribution)

Lane **S** of the **Swarm P3 Follow-ons Program** (`swarm/s`, base `64e191a` + the wave-0 program
ledger `245aef6`). Read the program ledger [`swarm-p3-ledger.md`](swarm-p3-ledger.md) first (charter,
frozen seams, fleet inventory, live substrate). This ledger records only lane S's deltas and consumes
the P1/P2-frozen seams verbatim — in particular the A3 `JoinCredentials`/`EngineParams` contract
([`swarm-ledger-p2-a3.md §2`](swarm-ledger-p2-a3.md)), which lane S extends **additively only**
(`#[serde(default)]`, back-compat), and the frozen node↔cloud presign HTTP contract
([`presign.rs`], `tests/fixtures/presign-*.json`, cloud `packages/shared/src/swarm/keys.ts`).

- **Repo / branch:** `daemon-node`, `swarm/s`, worktree `/home/j/experiments/daemon-worktree/p3-s`.
- **No cloud change.** The cloud DO is Lane R's; lane S consumes the existing presign/registry HTTP
  surface unchanged (the content-addressed keys ride the **frozen** `artifact` presign kind, below).
- **Frozen (respect, extend additively):** `tabi@1`, wire v42, the observe surface, the A3
  `JoinCredentials`/`EngineParams` contract, the presign DTO + R2 key layout, the guest guard.

## The problem (from the P2 gate's two scale caveats)

The P2 WAN gate ran tiny-llama with a **synthetic** corpus and **pre-staged** the experiment `.wasm`
onto every fleet box via `DAEMON_TRAIN_MODULE=<local path>` (gate runbook §1c–§2). The 160M envelope +
a real tokenized corpus were never distributed across the fleet — the root of the gate's ε and
overhead caveats. Lane S makes the fleet fetch **both** artifacts by content hash from the payload
store (SigV4 direct-to-R2), verify blake3 before use, and cache content-addressed — removing all
pre-staging. This is the staging half of the Merge-2 program gate.

## Design (as implemented)

### 0. Asset-generality decision (user, mid-lane — recorded)

A later wave will fetch the **CUDA runtime (nvrtc)** on demand like an asset, keyed by driver
version. Lane S's layout is deliberately **kind-agnostic** so that lands without reshaping anything:

- **On-disk cache:** [`ContentCache`] stores bare `objects/<hex-blake3>` — no module/corpus kind in
  the key. Any future asset (an nvrtc archive, a tokenizer, a checkpoint base) caches identically.
- **Store keys:** the R2 side rides the frozen `artifact` presign kind (`runs/<run>/<path>`); the
  per-kind prefixes (`modules/…`, `corpus/…`) are just path conventions — a future `assets/<blake3>`
  (or `assets/nvrtc/<driver>/<blake3>`) prefix is purely additive, no contract change.
- **Fetch machinery:** `fetch_cached` / `fetch_artifact_from_store` take an `ArtifactRef`
  (url + blake3) — nothing module-specific. The nvrtc fetcher itself is **NOT** built here (explicit
  non-goal); only the layout is kept generic.

### 0b. Lane-R coordination note (recorded mid-lane)

Lane R's live checkpoint-resync **reads the existing §11.3 keys** (`record-set.cbor`, round payloads,
`checkpoints/round-<r>.safetensors`) — no new payload-plane surface from S. **Constraint S honors:**
R2 lifecycle must retain **checkpoint objects for at least `payload_retention_rounds`** for
fleet-scale resync. Lane S compliance check: the [`ContentCache`] eviction is a **worker-local disk
cache** of immutable content-addressed artifacts (modules/shards) — it never deletes store-side
objects; the only store-side pruning in the tree is the pre-existing `FsPayloadStore::prune`, which
prunes **round-payload dirs only** (never `checkpoints/`). The actual R2 lifecycle rule is bucket
config (cloud side, Lane R / integration owner) — flagged here so the Merge-1/2 owner sets the bucket
lifecycle with the checkpoint-retention floor, not just the round-payload TTL.

### 1. Content-addressed on-disk cache — `daemon_swarm_net::content_cache::ContentCache`

A new additive module in `daemon-swarm-net` (the artifact-fetch crate). A blake3-keyed on-disk cache:

- **Layout:** `<root>/objects/<hex-blake3>` — one file per content hash. The hash IS the name, so a
  module/shard built once is cached once and shared across runs (the §6.1 "built, cached once"
  property holds on the worker side regardless of the per-run R2 key).
- **API:** `open(root, max_bytes)`, `contains(&Hash)`, `get(&Hash) -> Option<Vec<u8>>` (verifies on
  read; a corrupt cache file is treated as a miss + evicted), `insert(Hash, &[u8])` (blake3-checks the
  bytes match the key, writes atomically via a `.tmp` rename, then evicts).
- **Eviction policy (documented):** size-bounded **LRU by access time**. Each `get`/`insert` touches
  the file's mtime; when a new insert would exceed `max_bytes`, least-recently-used objects are
  removed until it fits. An object larger than the whole budget is fetched + returned but **not**
  cached (never evict everything for one giant object). The budget defaults to
  `[swarm].data_cache_gb` semantics (`from_gb`), overridable by `DAEMON_SWARM_CACHE_GB`. This mirrors
  the in-memory `ArtifactCache` (RUN-4) but persists across process restarts, so a fleet box warmed
  once never re-downloads.
- **fs discipline:** a scoped `#[allow(clippy::disallowed_methods)]` with the same justification as
  `artifact.rs::read_file_uri` — an operator/worker-controlled cache dir, and every byte is
  blake3-verified against its key on read, so a tampered cache file cannot smuggle a bad module.

### 2. Module distribution by content hash (worker assess path)

The authoring side pins the experiment module in the envelope artifact map at a **content-addressed
key**: `experiment.wasm = { url = "r2://modules/<blake3>.wasm", blake3 = <hash> }` (the run-relative
path `modules/<blake3>.wasm` → the frozen `artifact` presign kind → R2 key
`runs/<run>/modules/<blake3>.wasm`; no cloud change). The worker's `resolve_module` (backend.rs, gated
`#[cfg(feature = "swarm-net")]` since egress/presign live behind that feature) resolves it:

1. `DAEMON_TRAIN_MODULE` set → read the local file (the **explicit** dev/test override — the only
   remaining local-path path).
2. else, if the envelope module artifact is `r2://` / `https://` / `hf://` and the presign context
   env is present (`DAEMON_SWARM_PRESIGN_BASE`, `DAEMON_SWARM_RUN_ID`, `DAEMON_SWARM_ORG`/`_ACTOR`
   internal identity or `DAEMON_SWARM_BEARER`): check the `ContentCache` by the artifact's blake3 →
   hit returns the verified bytes; miss fetches via `ArtifactResolver::with_egress(...).with_presign(...)`
   (presigned GET, blake3-verified **before** the bytes are handed to `assess`/instantiation), then
   caches. The context is small env strings the node sets when spawning the worker — NOT a pre-staged
   GB artifact.
3. else → `file://` via the file-only resolver (unchanged).

`resolve_run` verifies the signed envelope and re-derives the module hash from the artifact map, so a
tampered module is rejected before the wasm engine ever loads it (§6.5, §12).

### 3. Corpus distribution (worker live path)

`EngineParams` gains an additive optional `corpus: Option<CorpusRef>`
(`CorpusRef { manifest_blake3, manifest_size, window_start, window_sequences }`,
`#[serde(default)]`). Absent ⇒ the existing `Corpus::synthetic` fallback (every pre-P3 caller + the CI
suites are byte-identical). Present ⇒ the worker (live.rs):

1. fetches the **manifest** object by content hash (`corpus/<manifest_blake3>.json` artifact key),
   blake3-verifies, parses `daemon_swarm_run::data::Manifest`;
2. computes the shards covering the run's **active data window** `[window_start, window_start +
   window_sequences)` (with wrap) via the new `Manifest::shards_covering` — so a peer fetches **only**
   the shards the run touches, not the whole (multi-GB) corpus (the host-RAM budget guard, spec §8 "shards
   download lazily ahead-of-need … LRU bounded"; M4 32 GB);
3. fetches each covering shard by content hash (`corpus/<shard_blake3>.bin`), blake3-verifies, caches
   in the `ContentCache`;
4. builds a **windowed** `Corpus` (new `Corpus::windowed`, sparse shards) that serves
   `sequence(batch)` from the resident shards and errors `ShardNotResident` if an unfetched shard is
   addressed (never a silent NaN).

The window `[window_start, window_sequences)` is authored by the run author (who knows
`rounds × global_batch`), so it is deterministic and identical across peers → digests agree. Per-round
seed-exact shard selection (fetch strictly the current round's assignment reactively at `RoundOpen`)
is a documented follow-on (it would touch Lane R's engine loop + needs the coordinator seed chain);
the active-window staging is the honest RAM-bounded windowing for the rehearsal + gate.

### 4. Publish side — `xtask` authoring/upload

- `xtask publish-module --module <path> --run <id> --presign-base <url> [auth]` — blake3s the module,
  presigns a PUT for `modules/<blake3>.wasm`, uploads via egress (direct-to-R2 on the SigV4 plane),
  prints the `blake3` + `size` + the `r2://modules/<blake3>.wasm` URL for the create-run request.
- `xtask publish-corpus --manifest <dir>/manifest.json --run <id> --presign-base <url> [auth]` — reads
  the `tokenize-corpus` output, uploads each shard to `corpus/<shard_blake3>.bin` and the manifest to
  `corpus/<manifest_blake3>.json`, prints the manifest hash/size + window hints. `tokenize-corpus`
  (M1) is unchanged; publish is a thin uploader on top.

### 5. Authoring wiring

`bins/swarm-local --emit-create-request` (+ the e2e fleet harness) author the module artifact at the
content-addressed `r2://modules/<blake3>.wasm` URL and (when a corpus dir is given) emit the
`CorpusRef` into the `JoinCredentials.engine`. The synthetic path stays the default when no corpus is
supplied.

## Files owned this lane (all additive)

- `crates/swarm/daemon-swarm-net/src/content_cache.rs` (NEW) + `lib.rs` export.
- `crates/swarm/daemon-swarm-run/src/data.rs` (windowed `Corpus`, `Manifest::shards_covering`).
- `crates/swarm/daemon-swarm-run/src/protocol.rs` (`ModuleRef`/`CorpusRef` additive; `EngineParams.corpus`).
- `crates/coprocessor/daemon-train/src/bin/daemon-train-worker/{backend.rs,live.rs}` (fetch wiring —
  coordinated with G [backend arm] + R [engine/rejoin]: distinct regions, additive).
- `xtask/src/{publish.rs (NEW),main.rs}` (publish paths).
- `bins/swarm-local/src/main.rs` (content-addressed authoring).
- `tests/daemon-swarm-e2e/tests/*` (env-gated fetch-by-hash rehearsal).
- `docs/specs/swarm-ledger-p3-s.md` (this) + `swarm-p3-artifact-distribution-runbook.md` (addendum).

## Planned commit slices (each green per the gates)

1. `mirror(S): ledger` — this file.
2. `feat(swarm-net): content-addressed on-disk cache (ContentCache) (green)`.
3. `feat(swarm-run): windowed corpus + shards_covering + additive CorpusRef/ModuleRef (green)`.
4. `feat(train-worker): fetch experiment module by content hash from the payload store (green)`.
5. `feat(train-worker): fetch assigned corpus shards by content hash; windowed corpus (green)`.
6. `feat(xtask,swarm-local): publish-module/publish-corpus + content-addressed authoring (green)`.
7. `mirror(S): ledger — results + 160M rehearsal + fleet staging matrix` (final).

## Gates

fmt; clippy workspace + feature combos (`daemon-train --features swarm-net`, `--features wgpu`,
`daemon-swarm-net --features ws,iroh`, `daemon-swarm-run --features iroh`) `-D warnings`; deny;
`cargo test --workspace`; net/run/train/e2e suites incl. the new fetch/verify/cache tests
(unit + env-gated live); build-guests before wasm tests; wasm32 (`daemon-swarm-{proto,coordinator}`);
typos. Known flake (never modify): the `daemon-conformance` detached-delegation trio +
`late_join_mid_run_syncs_and_contributes` — green-in-isolation is the standing disposition.

## Results

_(finalized below after the gates + the 160M local rehearsal + fleet staging.)_
