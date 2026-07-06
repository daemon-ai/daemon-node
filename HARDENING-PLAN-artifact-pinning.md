# HARDENING-PLAN — Phase 3 / Cluster E (artifact provenance pinning)

Worktree: `/home/j/experiments/daemon-worktrees/artifact-pinning`
Branch: `hardening/artifact-pinning` (off `hardening/integration`, which already contains all
Phase 1+2 work, incl. the Wave-1 shared SSRF egress client `crates/engine/daemon-egress`).

**Status: APPROVED (2026-07-05) — implementing. Scope: MODEL ARTIFACT PINNING ONLY, NODE-LOCAL,
ZERO WIRE CHANGE.**

Guiding principle (from the master plan): *make the unsafe form unrepresentable, not "remember to
check."* Today a downloaded model artifact is trusted after only a **size + `GGUF` 4-byte magic**
check — a supply-chain vector (a poisoned cache file, a tampered mirror). This track pins a
cryptographic **sha256** for downloaded single-file (GGUF) model artifacts, verifies it **before
load** (refusing on mismatch with `ModelError::Integrity`), and verifies the **download** against
the Hub-declared git-LFS `oid`.

---

## 0. APPROVED DECISIONS (supersede any conflicting detail below)

1. **D1 — TWO-LAYER, APPROVED.** L1: verify the *download* against the Hub-declared git-LFS `oid`
   (= sha256), which the tree API already returns — genuine source provenance at near-zero cost;
   **extend `RawLfs` to actually parse + use `oid`**. L2: persist that verified hash as a node-local
   pin and **re-verify it before every load, refusing on mismatch**.
2. **D2 — NODE-LOCAL SIDECAR, *not* a wire field.** Do **NOT** add `sha256` to `InstalledModel`; do
   **NOT** bump `WireVersion`; do **NOT** touch `daemon-api.cddl` or run codec regen. The security
   goal is entirely node-side in `resolve()`/`artifact_intact`. The pin lives in a **file beside the
   artifact** (`<local_path>.sha256`), which survives restart and is verified before load. Exposing
   the hash on the wire (`InstalledModel`) is deferred to the Phase-5 codec bundle (coordinator is
   tracking it).
3. **D3 — SKILL-BUNDLE SIGNING DEFERRED ENTIRELY to Phase 5.** It is wire-dependent (the signature
   rides on `SkillBundle` over NodeApi), optional, default-off. **This track does NOT touch
   `daemon-skills`, `import_bundle`, or `SkillBundle`.**
4. **D4 — ALWAYS-HASH-ON-LOAD, APPROVED** as the secure default. `resolve()` runs once per
   activation/switch (the local provider caches the built provider — `daemon-providers/src/local.rs:691-698`),
   not per turn, so the cost is bounded; size+mtime is spoofable and misses content swaps. A
   fast-path can be added later if a specific multi-GB model makes activation latency a real problem.
5. **D5 — moot** (skill signing deferred → no `bc-components`, no new dependency).

**NET:** ZERO wire change, no CDDL/codec churn, **no new dependency** (`sha2` is already a direct
`daemon-models` dep), no skills edits. Just model-artifact pinning in `daemon-models`.

---

## 1. Inventory (exact file:line)

- `crates/providers/daemon-models/src/hf/files.rs`
  - `RawLfs` L30-34 parses **only `size`** — the git-LFS `oid` (the sha256) is discarded. **Extend
    to parse `oid`.** `fetch_tree` L106-124; `to_model_file` L141-159; `list_files` L40-59.
- `crates/providers/daemon-models/src/resolve.rs`
  - `plan()` L22-95 builds `PlanFile`s — **where `expected_sha256` (the oid) is threaded in** (llama
    single-file path L37-81; mistral dir path L82-93 left unpinned).
- `crates/providers/daemon-models/src/acquire.rs`
  - `PlanFile` L34-44 (internal, non-wire) — add `expected_sha256: Option<String>`.
  - `ResolvedArtifact` L67-76 (internal) — add `sha256: Option<String>` (the pin value).
  - `run_job()` L381-502 — size check L446-462; **add L1 hash verify + capture the primary pin
    after the size check** (in `spawn_blocking`).
- `crates/providers/daemon-models/src/manager.rs`
  - `resolve()` L332-379 — **add verify-before-load** (refuse on pin mismatch) in the cataloged
    fast-path L352-368; the `Local`/download-return `ResolvedArtifact` literals L340/L362/L374 gain
    `sha256: None`.
  - `catalog_artifact()` L456-470 — **write the sidecar pin** from `artifact.sha256`.
  - `artifact_intact()` L530-546 — unchanged (benign exists/size/magic → re-acquire on false).
  - `best_effort_delete()` L603-609 — also remove the `<path>.sha256` sidecar.
- `crates/providers/daemon-models/src/{hash.rs|gguf.rs}` — new `sha256_file(path)` streaming helper
  (uses the already-present `sha2`).
- `crates/providers/daemon-models/src/error.rs` — `ModelError::Integrity(String)` L42-44 **already
  exists** → the refuse-on-mismatch error (reused; no new variant). `ModelError::Download` for the
  L1 download failure.
- `crates/providers/daemon-providers/src/local.rs:691-705` — `resolve` is called once per switch
  (cached otherwise) → bounds load-time hashing (D4).
- **NO wire type touched:** `InstalledModel` (`daemon-common`), `daemon-api.cddl`, fixtures, and the
  vendored C codec are all left unchanged. `cargo test -p daemon-api --features arbitrary` is run
  anyway to **prove zero wire drift**.

---

## 2. Design — node-local sidecar pin, two layers

### 2a. L1 — download-time verification against the Hub `oid`  *(source provenance)*
- Extend the internal `RawLfs` to parse `oid` (git-LFS sha256; every GGUF is LFS). Add a
  `pub(crate) list_files_with_oids()` in `hf/files.rs` returning `(Vec<ModelFile>, HashMap<path,oid>)`
  from a single `fetch_tree`; `list_files` delegates to it and drops the oids (other callers
  unchanged). `resolve::plan` (llama path) populates `PlanFile.expected_sha256` from that map.
- In `run_job`, **after** the existing size check, when `file.expected_sha256.is_some()`: compute the
  downloaded file's sha256 (`spawn_blocking`) and compare. On mismatch → `fail(&job, "<file>: sha256
  mismatch — expected <oid>, got <hash> (tampered or corrupted download)")` → job `Failed`, artifact
  **never cataloged**. Absent oid → skip L1 (TOFU pin still recorded).

### 2b. L2 — record the pin + verify before load  *(the core gate)*
- **Record:** the primary single-file artifact's sha256 (the verified oid, or a TOFU computation
  when the Hub reported no oid) rides up on `ResolvedArtifact.sha256`. `catalog_artifact` writes it
  to `<local_path>.sha256` (best-effort; a failed write logs a warning and leaves the model
  loadable-but-unpinned, matching legacy — never blocks the install). Directory (mistral.rs)
  artifacts get no pin.
- **Verify before load:** in `resolve()`, before the cataloged fast-path returns, when a
  `<local_path>.sha256` sidecar exists and `local_path` is a file: recompute the file's sha256
  (`spawn_blocking`) and compare.
  - **match** → proceed (then `artifact_intact` does its cheap size/magic check).
  - **mismatch** → `Err(ModelError::Integrity("<display_name>: on-disk artifact sha256 does not
    match the pinned <pin> — refusing to load a tampered/corrupted model"))`. **Hard refusal — never
    silently re-download.**
  - **no sidecar** (legacy install) or **missing file** → `Ok(())`; `artifact_intact` then decides
    fast-path vs benign re-acquire (unchanged behavior).

### 2c. Helper
- `sha256_file(path) -> io::Result<String>` (buffered streaming read → lowercase hex), in a new
  `src/hash.rs`. Wrapped in `spawn_blocking` at every async call site.

---

## 3. Skill-bundle signing — DEFERRED (D3)

Out of scope for this track. No `daemon-skills` / `import_bundle` / `SkillBundle` edits. It lands in
the Phase-5 codec-regen bundle with the other deferred wire changes.

---

## 4. Wire-format impact — NONE (proven, not assumed)

No wire type changes. `cargo test -p daemon-api --features arbitrary` is run to **prove** zero
drift with no `daemon-api.cddl` edit. No `just update-codec` / `just codec-drift` needed. No new
dependency → `cargo deny` unaffected (run only if a dep is somehow added).

---

## 5. Tests — written FIRST, confirmed failing pre-fix

- **`tampered_pinned_artifact_is_refused_before_load`** (`manager.rs` unit) — seed a registry record
  for a GGUF file; write a matching `<path>.sha256` sidecar; then flip a byte (size + `GGUF` magic
  unchanged, so today's `artifact_intact` still trusts it). `manager.resolve(model)` →
  `Err(ModelError::Integrity(_))`. *Pre-fix:* `resolve` returns `Ok` (loads tampered file) → **fails**.
- **`valid_pinned_artifact_loads`** (`manager.rs` unit) — same, byte untouched → `resolve` → `Ok`
  with the right path (positive guard; no network).
- **`download_oid_mismatch_fails_and_is_not_cataloged`** (`acquire_mock.rs`, mock Hub) — serve a tree
  whose `lfs.oid` does **not** match the served bytes → job `Failed` (error contains "sha256
  mismatch"), catalog empty. *Pre-fix:* oid ignored → `Completed` → **fails**.
- **`download_oid_match_records_pin`** (`acquire_mock.rs`) — matching oid → `Completed`, the
  `<local_path>.sha256` sidecar exists and equals the oid. (Positive; also proves the pin is sourced
  from the oid.)

Regression: existing `daemon-models` tests (`acquire_mock`, `hf_mock`, `network`, `registry`,
`manager`) stay green — trees with no `lfs.oid` skip L1 and record a TOFU pin (download still
completes); the new `ResolvedArtifact.sha256`/`PlanFile.expected_sha256` fields get `None` at their
existing literals.

---

## 6. Gate (from worktree root, tails inspected)
```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo test -p daemon-api --features arbitrary      # PROVE zero wire drift (no CDDL edit)
```
- No new dep → `cargo deny` unaffected (run only if a dep is somehow added).
- **Machine-load note:** if `bins/daemon/tests/host_launch.rs` fails under the full parallel run,
  re-run isolated `cargo test -p daemon --test host_launch -- --test-threads=2` before treating as
  real. **Known pre-existing flakes (do not chase):** `detached_delegation` ×2, `process_notify`
  store-seam.
- Do NOT merge; do NOT remove the worktree.

---

## 7. Residuals / follow-ons (surfaced)
- **Directory (mistral.rs) artifacts are not pinned** — only single-file GGUF. A per-file manifest
  hash for snapshot dirs is a follow-on; today's existence check remains for them (no regression).
- **Split-GGUF sets:** the pin covers the primary (first-shard) `local_path` file, matching the
  existing `artifact_intact` scope; extra shards are not individually pinned (pre-existing gap).
- **TOFU when the Hub reports no `oid`** — the pin is the hash-of-what-we-downloaded; defends at-rest
  tampering (L2) but not a malicious first download. Real GGUF repos are LFS (always have `oid`).
- **Co-located sidecar** — a local attacker who rewrites the artifact could also rewrite
  `<path>.sha256`. L1 (source oid at download) is the strong provenance; L2 catches accidental /
  naive at-rest corruption and content-swaps that preserve size+magic. A tamper-resistant pin store
  is a follow-on.
- **Wire exposure of the hash** (`InstalledModel.sha256`) deferred to the Phase-5 codec bundle (D2).

---

## 8. Change summary (files touched)
- `crates/providers/daemon-models/src/hf/files.rs` — parse LFS `oid`; `list_files_with_oids`.
- `crates/providers/daemon-models/src/resolve.rs` — populate `PlanFile.expected_sha256`.
- `crates/providers/daemon-models/src/acquire.rs` — `PlanFile.expected_sha256`; `ResolvedArtifact.sha256`;
  L1 verify + pin capture in `run_job`.
- `crates/providers/daemon-models/src/manager.rs` — verify-before-load in `resolve`; write sidecar in
  `catalog_artifact`; remove sidecar in `best_effort_delete`; `sha256: None` on the other
  `ResolvedArtifact` literals.
- `crates/providers/daemon-models/src/hash.rs` — new `sha256_file` (uses existing `sha2`).
- Tests: `crates/providers/daemon-models/src/manager.rs` (unit) + `.../tests/acquire_mock.rs`.
- **No** `daemon-common` / `daemon-api.cddl` / fixture / codec / `daemon-skills` / `Cargo.toml` change.
