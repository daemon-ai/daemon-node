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

### Commit list (`swarm/s`, base `64e191a` + wave-0 `245aef6`; oldest → newest)

| Commit | Subject |
|---|---|
| `979d245` | `mirror(S): ledger` |
| `68b2aad` | `feat(swarm-net): content-addressed on-disk cache (ContentCache) (green)` |
| `b4afa83` | `feat(swarm-run): windowed corpus + shards_covering + additive CorpusRef (green)` |
| `163759d` | `feat(train-worker): fetch experiment module + corpus shards by content hash from the payload store (green)` |
| `6160148` | `feat(xtask): publish-module/publish-corpus — content-addressed artifact upload (green)` |
| `03966da` | `test(swarm-net): end-to-end fetch-by-hash + content-cache-serve integration test (green)` |
| `7c8b279` | `feat(train-worker): DAEMON_TRAIN_PREFETCH fleet cache-warming mode + asset-generality ledger note (green)` |
| `156b229` | `feat(train-worker): DAEMON_TRAIN_BACKEND engine selection + roomy 160M budgets (green)` |
| `3afec18` | `test(swarm-e2e): 160M fetch-by-hash staging rehearsal harness + WS observer message-log tap (green)` |
| `3dd75ec` | `fix(swarm-e2e): observer tap stop signal + bounded teardown; ledger: Lane-R checkpoint-retention note` |
| `ea883ef` | `docs(specs): P3 artifact-distribution runbook — content-addressed fetch replaces pre-staging` |
| (final) | `mirror(S): ledger — results + 160M rehearsal + fleet staging matrix` |

### Published artifacts (swarm-dev R2, SigV4 plane, asset scope `assets-p3s`)

- **Module** (tiny-llama guest = the 160M preset module):
  `r2://modules/86aa9cdcb0656e51f0ce0b2883adfc14274599b901d8f4ff285effbeaad0fddb.wasm`,
  blake3 `86aa9cdc…0fddb`, **143 960 B** (key `runs/assets-p3s/modules/<hash>.wasm`).
- **Corpus** — real **TinyStories** (`roneneldan/TinyStories@main`, `TinyStories-valid.txt`),
  **GPT-2 BPE** tokenizer, `seq_len 1024`, u16 shards: **4 723 712 tokens = 4613 sequences, 5 shards**
  (4 × 2 097 152 B + 1 × 1 058 816 B) at `r2://corpus/<shard-blake3>.bin`; manifest
  `r2://corpus/91b7eec22b5250328a94deacd60c2ae9463e0968c61c9c2e0c660e32e18c0f90.json`
  (blake3 `91b7eec2…c0f90`, 1039 B). Shard hashes: `7b7b9549…`, `1e0d8763…`, `5c7a372b…`,
  `8c7110fe…`, `c0f407c3…` (full values in the fleet matrix logs / manifest).

### THE 160M LOCAL REHEARSAL — EXECUTED GREEN (the lane exit criterion)

**Run `run-s-160m-1784024914`** (2026-07-14; the fully-green third execution — see deviations for
the first two): **2 local `daemon-train-worker` subprocesses** (debug, `swarm-net,wgpu`,
`DAEMON_TRAIN_BACKEND=wgpu` → **Vulkan/RADV** on Strix Halo), real Cloudflare coordinator
`daemon-swarm-dev` + **SigV4 real-R2**, the **full `llama_160m` preset (151 862 784 params,
vocab 50257, seq 1024)**, 3 rounds, `steps_per_round 2 × micro_batch 1`, `global_batch 4`,
`CorpusRef{window 0..12}` — **no `DAEMON_TRAIN_MODULE`, no pre-staged bytes anywhere**:

- **Module fetch-by-hash at assess:** both peers resolved `r2://modules/86aa…wasm` via presigned GET,
  blake3-verified before instantiation; assess (incl. the real 160M meta pass) ~96 s/peer.
  Eligibility on the unified pool: "fits at micro_batch=64 (~40.5 GiB device + ~1.2 GiB host,
  budget 112 GiB)".
- **Corpus fetch-by-hash at join:** manifest + **only the window's shard** staged
  (`shards_covering(0,12)` → shard 0 of 5 — the RAM-bounded windowing), blake3-verified, cached.
- **Per-round det digests — byte-identical across both peers, all 3 rounds:**
  ```text
  round 0: eb8168ab5590c4984bf1ef2615b32183  ×2
  round 1: 7b091dc875c26275cf46624c60498c93  ×2
  round 2: 6c96b46a0747d0f49e2ad9c0d53d78f2  ×2
  ```
- **Real 160M training on real TinyStories tokens:** per-step losses 10.85 → 9.85 over the 3 rounds
  (2 inner steps each), monotonically decreasing. Final DO state `phase=finished, round=3`.
- **Observe capture (worker-subprocess loop — the carried follow-on 3, message-log half):** a passive
  `WsControlPlane` tap wrote `/tmp/stage-160m-observe/run-s-160m-1784024914.dsmlog` (32 signed
  messages, rounds 0–2); **offline** verification from the artifact alone: dsmlog round-trips,
  `digest_tally_from_log` → `reporters=2 agreed=true outliers=[]` for every round, `RunHealth`
  projects 3 rounds. Console log archived beside it. (The engine-input `.dsmcap` half still rides
  the `swarm-local` harness path — the full-replay worker-loop capture remains a follow-on.)
- Harness: `tests/daemon-swarm-e2e/tests/staging_160m.rs`, test wall 658 s
  (`test result: ok. 1 passed`).

Two additional executions of the same envelope trained green earlier the same day
(`run-s-160m-1784020704`, `run-s-160m-1784023792` — identical round-0 digest `eb8168ab…`, 3/3 rounds
byte-identical within each run) but hung in the harness **teardown** (not the protocol); fixed by
`3dd75ec` (observer stop signal, capture-before-teardown, bounded Leave/shutdown timeouts).

### Fleet pre-staging matrix (Merge-2 pre-staging) — ALL FOUR BOXES GREEN

`DAEMON_TRAIN_PREFETCH` (the cache-warming mode) executed on every fleet box against the store:
**7 objects each (module + manifest + 5 shards), blake3 byte-identical on every box**, all fetched
`source=store` then verified. Cache = `ContentCache` (`objects/<hex-blake3>`, kind-agnostic).

| Peer | Worker binary (fingerprinted) | Cache dir | Per-object fetch (ms) | Total |
|---|---|---|---|---|
| **Strix Halo** (Linux/RADV, local) | local debug `swarm-net,wgpu` build | `~/.cache/daemon-swarm-content` | module 1263 · manifest 600 · shards 733/637/435/398/1373 | ~5.4 s |
| **M4 Mac** (Darwin arm64/Metal) | on-box build `~/daemon-node-p3s` (`swarm-net`, warm-cloned target, 17 s incremental) | `~/.cache/daemon-swarm-content` | 261 · 48 · 212/336/221/219/75 | ~1.4 s |
| **RunPod 4090** (Linux container) | on-box rebuild `/root/daemon-node-c3` (`swarm-net`; freshness fingerprinted: 4 `DAEMON_TRAIN_PREFETCH` strings, 2003 `tungstenite` strings — the P2 artifact-drift check) | `/root/.cache/daemon-swarm-content` | 385 · 144 · 238/144/178/242/155 | ~1.5 s |
| **Windows 5090** (Server 2022, cmd.exe over ssh) | sealed MinGW cross-build `.#daemon-train-worker-windows` (22 053 376 B), deployed `daemon-train-worker-p3s.exe`, **SHA256 `b7a19455…59d4` verified local == remote** | `%USERPROFILE%\daemon-swarm-content` | 229 · 74 · 104/212/197/194/98 | ~1.1 s |

**Idempotence proven** (Windows re-run): all 7 objects `source=cache`, 0–3 ms each — a warm box
never re-downloads. Cross-platform distribution incl. Windows cmd.exe and the pod: **proven**.
The 160M fleet measurement itself is **NOT run here** — that is Merge-2's ceremony; everything it
needs is staged + verified.

Fleet hygiene notes: the pod's disk was 94 % full — reclaimed P2's stale `/root/daemon-node-c3/target`
(a P2-runbook §9 cleanup item) before the on-box rebuild; Lane G's `/root/daemon-node-p3g` untouched.
M4 build used an APFS clone of the P2 target for a 17 s incremental build; new dir `~/daemon-node-p3s`.

### Gate results (final HEAD; jobs capped ≤ 16 = nproc/2, one build at a time)

Recorded from the final gate sweep (see the terminal log `/tmp/p3s-final-gates.log`):

- `cargo fmt --all --check` ✓ · `cargo clippy --workspace --all-targets -- -D warnings` ✓.
- Feature-combo clippy `-D warnings`: `daemon-train --features {swarm-net, swarm-net+wgpu,
  burn-ndarray}` ✓ · `daemon-swarm-net --features ws,iroh` ✓ · `daemon-swarm-run --features iroh` ✓ ·
  `daemon-swarm-e2e --features iroh` ✓ (and default) — run per-slice during the lane, re-verified at
  HEAD via the workspace sweep.
- `cargo deny check` ✓ (xtask's new deps are existing workspace entries — lock adds no third-party
  crate; the e2e `iroh` feature now also pulls `daemon-swarm-net/ws`, already in the lock).
- `cargo test --workspace`: green mid-lane (full run at `7c8b279`); at final HEAD the **only**
  failures were the documented pre-existing `daemon-conformance` detached-delegation flake
  (`detached_fanout_materializes_distinct_children` — crate untouched by this lane, `git diff` empty;
  nondeterministic this session even single-threaded: 2-of-3 green alone, the same documented flake
  class → standing green-in-isolation disposition, flagged for the run-crate owner) — no swarm suite
  failed.
- Swarm suites: `daemon-swarm-net --features ws,iroh` ✓ (incl. content_cache 7/7 + the fetch-by-hash
  integration test) · `daemon-swarm-run --features iroh` ✓ (45 incl. windowing + CorpusRef
  back-compat) · `daemon-train --features burn-ndarray` ✓ · `daemon-swarm-e2e` default ✓ ·
  `live_transport --features iroh` **7/7** ✓ in isolation (one parallel-load flake under the full
  sweep, green alone — the standing rule).
- `cargo run -p xtask -- build-guests` ✓ · wasm32 builds (`daemon-swarm-{proto,coordinator}`) ✓ ·
  `typos docs/specs` ✓.
- **The 160M staging rehearsal EXECUTED GREEN in-session** (the headline above).
- Known flake standing rule applies (`daemon-conformance` detached trio — green-in-isolation).

### Deviations (recorded honestly)

1. **The reduced-preset CPU validation run stalled** (`run-s-160m-1784019579`): both workers assessed
   + joined green (module fetch-by-hash worked) but committed no round within the budget — the
   160M-*shaped* reduced model (d_model 256, vocab 50257, seq 1024) on the **CPU det lane in a debug
   build** is minutes/step; the DO aged the round out. Not a distribution failure; superseded by the
   Vulkan runs (the rehearsal criterion). Lesson recorded: size rehearsal presets to the backend.
2. **Harness teardown hang** (first two full runs): `Leave` on a 160M wgpu worker + an observer
   subscription with no close signal hung the harness *after* all assertions passed. Fixed
   (`3dd75ec`); run 3 is the clean end-to-end record.
3. **`DAEMON_SWARM_RUN_ID` re-scopes the artifact presign** (worker): content-addressed assets are
   published once under a shared scope (`assets-p3s`) and consumed by many runs — the presign
   endpoint scopes keys without requiring a live run. The corpus fetch honors the same override, so
   module + corpus resolve under one scope. A cleaner authored-in-credentials scope field is a
   possible Merge-1 refinement (env is the node-sets-worker-env convention today).
4. **Observe is the message-log half.** `<run>.dsmlog` from the worker-subprocess loop is captured +
   offline-verified (digest tally, run health); the `.dsmcap` engine-input capture (full byte replay)
   still rides the `swarm-local` harness path — carried.
5. **`swarm-local --emit-create-request` authoring not extended** — the rehearsal harness authors the
   content-addressed envelope directly (same seam). Folding `--module-hash`/`--corpus-manifest` flags
   into `bins/swarm-local` is a small Merge-1 follow-on if the ceremony wants CLI authoring.

### Seams lane S exports (freeze at merge; all additive)

- **`daemon_swarm_net::ContentCache`** — blake3-keyed on-disk cache: `open`/`open_gb`, `contains`,
  `get` (verify-on-read, poisoned-entry eviction), `insert` (hash-checked, atomic, LRU-evicting).
  Layout `objects/<hex-blake3>`, **kind-agnostic** (the nvrtc-as-asset decision, §0).
- **`daemon_swarm_run::protocol::CorpusRef`** + `EngineParams.corpus: Option<CorpusRef>`
  (`#[serde(default)]`, back-compat proven by test).
- **`daemon_swarm_run::data`**: `Corpus::windowed`, `Corpus::resident_shards`,
  `Manifest::shards_covering`, `DataError::{ShardIndexOutOfRange, ShardNotResident}`.
- **Worker env contract**: `DAEMON_SWARM_{PRESIGN_BASE,RUN_ID,ORG,ACTOR,BEARER,CACHE_DIR,CACHE_GB}`
  (fetch context), `DAEMON_TRAIN_BACKEND` (cpu|burn-ndarray|wgpu + roomy budgets),
  `DAEMON_TRAIN_PREFETCH{,_MODULE,_MANIFEST,_WINDOW}` (cache warming). `DAEMON_TRAIN_MODULE` demoted
  to the explicit dev/test override.
- **xtask**: `publish-module`, `publish-corpus` (presigned PUT, content-addressed keys
  `modules/<blake3>.wasm`, `corpus/<blake3>.{json,bin}` under the frozen `artifact` presign kind —
  no cloud change).
- **e2e**: `staging_160m.rs` (env-gated `SWARM_STAGE_*`), the WS observer message-log tap;
  `daemon-swarm-e2e/iroh` now also enables `daemon-swarm-net/ws` (dev-dep feature, lane-owned).
- **Docs**: `swarm-p3-artifact-distribution-runbook.md` (replaces the P2 §1c–§1e artifact scp steps).

### What Merge-1 / Merge-2 must know

- **Merge-2's ceremony switches peer specs from `DAEMON_TRAIN_MODULE=<path>` to the `DAEMON_SWARM_*`
  fetch env** (runbook addendum §3–§4). All four fleet caches are warm for the published 160M module
  + TinyStories corpus (hashes above) — the measurement run consumes them as `source=cache`.
- **Corpus sizing for the measurement run:** the published TinyStories-valid corpus is 4613 sequences
  (5 shards). A longer 160M measurement (more rounds × bigger global_batch) should publish the full
  TinyStories train split with the same commands (content-addressed → additive; the window math
  already stages per-box subsets).
- **Lane-R composition:** resync reads the existing §11.3 keys — nothing here collides; the
  **bucket lifecycle must retain checkpoint objects ≥ `payload_retention_rounds`** (§0b — an
  integration-owner/cloud config item, not code in this lane). The dev coordinator was NOT redeployed
  by S (no cloud change); if R redeploys with the checkpoint-pointer, the presign/artifact surface S
  uses is unchanged (verified additive).
- **`Cargo.lock`** gained no third-party crate (xtask/e2e edits reference existing workspace deps).
  `guests/guests.blake3` untouched (warn-and-rebuild guard; no guest source change).
- **The `assets-p3s` scope convention**: published assets live under `runs/assets-p3s/…` on the
  swarm-dev bucket. A production naming decision (e.g. a dedicated `assets/` prefix outside `runs/`)
  is a cloud-side key-layout choice for a later wave — the frozen `artifact` kind carries it either
  way (asset-generality note §0).
