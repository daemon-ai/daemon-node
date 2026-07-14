# Swarm P3 — artifact distribution runbook (replaces the P2 pre-staging steps)

P3 lane-S addendum to [`swarm-p2-gate-runbook.md`](swarm-p2-gate-runbook.md). It **replaces the
runbook's §1c–§1e module pre-staging** (`scp tiny_llama.wasm <box>` + `DAEMON_TRAIN_MODULE=<path>`)
and the synthetic corpus with **content-addressed fetch from the payload store**: every fleet peer
pulls the experiment module and its assigned corpus shards by blake3 from R2 (presigned GET,
SigV4 direct-to-R2), verifies before use, and caches on disk. The worker binary still deploys per
the P2 runbook (§1b–§1e build/cross-build steps are unchanged); what no longer travels by `scp` is
the **artifacts** — module + corpus. P2's lesson ("never trust pre-staged artifacts") becomes
structural: nothing is pre-staged, everything is hash-verified at fetch.

Evidence for every step below: [`swarm-ledger-p3-s.md`](swarm-ledger-p3-s.md) (the 160M local
rehearsal + the fleet staging matrix).

## 0. One-time publish (run author, any box with the artifacts)

Artifacts are published **once** under a shared asset scope (`--run <scope>`, e.g. `assets-p3s` —
the presign endpoint scopes keys `runs/<scope>/…` and does not require the scope to be a live run;
content-addressed objects are consumed by many runs):

```bash
# Module → r2://modules/<blake3>.wasm  (prints the blake3 + size for the envelope [artifacts])
nix develop --command cargo run -p xtask -- publish-module \
  --module guests/target/wasm32-unknown-unknown/release/tiny_llama.wasm \
  --run assets-p3s \
  --presign-base https://daemon-swarm-dev.me-dc6.workers.dev/api/v1/swarm \
  --org org_live --actor key:live      # (or --bearer <token> on the gateway path)

# Corpus: tokenize offline (M1 seam), then publish shards + manifest by content hash
nix develop --command cargo run -p xtask -- tokenize-corpus \
  --dataset roneneldan/TinyStories --dataset-file TinyStories-valid.txt --revision main \
  --tokenizer gpt2 --out-dir /tmp/ts-corpus-160m --seq-len 1024 --shard-tokens 1048576
nix develop --command cargo run -p xtask -- publish-corpus \
  --manifest /tmp/ts-corpus-160m/manifest.json --run assets-p3s \
  --presign-base https://daemon-swarm-dev.me-dc6.workers.dev/api/v1/swarm \
  --org org_live --actor key:live
# → shards at r2://corpus/<shard-blake3>.bin, manifest at r2://corpus/<manifest-blake3>.json;
#   prints CorpusRef.{manifest_blake3, manifest_size} + total sequences.
```

Publishing is idempotent (content-addressed keys re-PUT the same bytes). The publisher verifies
each shard's blake3 against the manifest before upload and uploads the manifest **last** (a partial
corpus never has a resolvable manifest).

## 1. Run authoring (what changes vs P2 §3)

- **Envelope `[artifacts]`:** the module rides its content-addressed URL —
  `"experiment.wasm" = { url = "r2://modules/<blake3>.wasm", blake3 = <hash> }` — and the corpus
  manifest is a named artifact — `"data.manifest" = { url = "r2://corpus/<blake3>.json", blake3 = … }`
  with `[data].manifest = "data.manifest"`.
- **`JoinCredentials.engine.corpus`** (additive `CorpusRef`): `manifest_blake3` + `manifest_size` +
  the active window `window_start`/`window_sequences` (= `rounds × global_batch`; `0` = whole
  corpus). Present ⇒ workers fetch the real corpus; absent ⇒ the synthetic fallback (tests only).
  `corpus_vocab_clamp = 0` for a real corpus tokenized at the model's vocabulary.
- **Backend:** `DAEMON_TRAIN_BACKEND=wgpu` (or `cpu`/`burn-ndarray`) in the worker env selects the
  training backend + the roomy 160M sandbox budgets. Det digests are backend-independent (B3).

## 2. Worker-side resolution (automatic; what to configure)

At **assess**, the worker resolves the envelope module artifact: `DAEMON_TRAIN_MODULE` (explicit
dev/test override — the ONLY remaining local-path route) → else content cache → else presigned GET,
blake3-verified **before instantiation**. At **join**, `engine.corpus` triggers the manifest fetch +
staging of exactly the shards the active window touches (`Manifest::shards_covering` — the M4-32GB
RAM guard), each blake3-verified, then a windowed corpus (addressing an unstaged shard is a typed
error, never a silent read).

The fetch context is plain env (the node sets it at spawn; on Windows cmd.exe:
`set VAR=… && daemon-train-worker.exe`):

| Env | Meaning |
|---|---|
| `DAEMON_SWARM_PRESIGN_BASE` | coordinator presign base |
| `DAEMON_SWARM_RUN_ID` | asset scope for artifact keys (`runs/<scope>/…`); defaults to the joined run |
| `DAEMON_SWARM_ORG`/`DAEMON_SWARM_ACTOR` or `DAEMON_SWARM_BEARER` | presign auth |
| `DAEMON_SWARM_CACHE_DIR` (default `<tmp>/daemon-swarm-cache`), `DAEMON_SWARM_CACHE_GB` (default 20) | the on-disk content cache |

**Cache layout (asset-kind-agnostic by design):** `<cache>/objects/<hex-blake3>` — one file per
content hash, verified on every read (a corrupt file is evicted + refetched), atomic writes,
size-bounded LRU-by-write-time eviction. The key carries **no kind**: modules, shards, and any
future asset (e.g. the driver-keyed nvrtc runtime, a later wave) cache identically. R2-side
retention: the cache never deletes store objects; note the Lane-R constraint — the bucket lifecycle
must retain **checkpoint** objects ≥ `payload_retention_rounds` (see `swarm-ledger-p3-s.md` §0b).

## 3. Fleet cache warming (replaces P2 §1c–§1e artifact scp)

Each fleet box runs the worker's **prefetch mode** once — fetch + verify + cache, print evidence,
exit. Idempotent (a warm box reports `source=cache`).

```bash
DAEMON_TRAIN_PREFETCH=1 \
DAEMON_SWARM_PRESIGN_BASE=https://daemon-swarm-dev.me-dc6.workers.dev/api/v1/swarm \
DAEMON_SWARM_RUN_ID=assets-p3s DAEMON_SWARM_ORG=org_live DAEMON_SWARM_ACTOR=key:live \
DAEMON_SWARM_CACHE_DIR=$HOME/.cache/daemon-swarm-content \
DAEMON_TRAIN_PREFETCH_MODULE=<module-blake3> \
DAEMON_TRAIN_PREFETCH_MANIFEST=<manifest-blake3> \
DAEMON_TRAIN_PREFETCH_WINDOW=<start>:<count>   # optional; absent/0 = every shard \
  ./daemon-train-worker
```

Windows cmd.exe over ssh (one line): `set DAEMON_TRAIN_PREFETCH=1 && set DAEMON_SWARM_… && daemon-train-worker.exe`.
Requires a worker built with `--features swarm-net` (the fail-fast rule from P2 Merge-3 applies:
a swarm-net-less build errors loud). Record each box's printed `blake3/bytes/source/ms` lines in
the ledger — that is the staging evidence Merge-2 consumes.

## 4. The rehearsal / measurement run (P2 runbook §2–§7 otherwise unchanged)

Drive the run with `tests/daemon-swarm-e2e/tests/staging_160m.rs` (2-peer local rehearsal — the
lane-S exit criterion, executed green; see the ledger) or the P2 ceremony harness with the
content-addressed envelope. Peer spec change vs P2 §2: remote peers **no longer bake
`DAEMON_TRAIN_MODULE`** into the ssh command — they carry the `DAEMON_SWARM_*` fetch context
instead. The observer tap writes `<run>.dsmlog` (message-log capture; offline digest tally +
run health) alongside the digest transcript.
