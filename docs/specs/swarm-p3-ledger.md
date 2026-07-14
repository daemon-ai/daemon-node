# Swarm P3 Follow-ons — program ledger

Wave-0 scaffold coordination record for the **Swarm P3 Follow-ons Program** — the direct-measurement
follow-on to the completed **Swarm P2 WAN Program** ([swarm-p2-ledger.md](swarm-p2-ledger.md)). P2
passed the spec §17 WAN research gate with **two honest scale caveats** (ε-convergence and <15% round
overhead), both rooted in the same gap: the **160M envelope + tokenized corpus were never staged onto
the fleet**. P3 converts those two caveats into direct measurements, and lands the two carried engine
follow-ons that make the measurement run the strongest end-to-end evidence the stack has produced: the
**CUDA engine arm** and **live checkpoint-resync**.

This is the single source of truth for the P3 trunk, lane file-ownership, the frozen-file rule, the
inherited conventions, the live substrate endpoints, and the fleet inventory. Lane agents working in a
P3 worktree: **read this before you touch anything** — it carries everything you need without reaching
into `~/.cursor`.

This ledger governs P3 on top of the completed P1+P2 programs. The P2 program record
(`swarm-p2-ledger.md`) — its Merge-1/2/3 **frozen-interfaces** sections and the **P2 WAN gate
ceremony** evidence — and the P1 record (`swarm-p1-ledger.md`) remain authoritative for every seam
frozen at or before the P2 exit. **P3 inherits every P1+P2 frozen seam: extend them additively only.**
In particular `tabi@1` (the 66-op tensor ABI) is **FROZEN FOREVER** (spec §16) — additive `op@version`
growth only; a breaking change is `tabi@2` — and the wire stays at **v42**.

## Base + trunk

- **Repo:** `daemon-node` (Rust backend submodule; standalone checkout).
- **Base commit:** `64e191a` (`docs(config): regenerate config-reference for swarm registry keys`) —
  daemon-node master, the landed P2 WAN program (P2 trunk `integrations/swarm-p2` merged to master).
  daemon-cloud `daemon-api` master = `b13f51d` (the P2 `swarm/p2-integration` landing; deployed
  coordinator version `95cbb0f1` still matches it).
- **Trunk:** `integrations/swarm-p3` (one shared trunk, forked from `64e191a`), worktree at
  `/home/j/experiments/daemon-worktree/swarm-p3-integration`. G/R/S lanes merge back here; the
  integration owner owns the frozen files, any (unexpected) wire bump, seam swaps, and this ledger.
- **Worktrees:** each lane subagent works EXCLUSIVELY in its assigned worktree under
  `/home/j/experiments/daemon-worktree/` (daemon-node) or a branch checkout of
  `/home/j/experiments/daemon-cloud/daemon-api` (Lane R's small cloud half). Never modify the main
  checkouts, never `git push`, never `--no-verify`.

### Wire version note (READ THIS)

`WireVersion::CURRENT` is **v42** on this base (the P2 additive `SwarmHardwareReport.shared_mb`
surface; `contract_wire_version_is_v42`). **No P3 wire change is expected** — the three lanes are
engine/runtime/staging work, not SwarmApi surface. Lane R's checkpoint-pointer is a **cloud DO +
node-runtime** contract (payload-store + coordinator surface), not a SwarmApi wire type; telemetry
stays off-wire (rides `SwarmEvent::Warning` classes). If a wire change becomes unavoidable it is
additive, targets **v43**, and is a **single coordinated integration-owner commit** at the merge that
introduces it (mirroring the P2 41→42 discipline), with `just update-codec` + `codec-drift` green (the
superproject codec regen is a human, signed step).

## Program charter — the three lanes (from the program plan, verbatim-ish)

Trunk `integrations/swarm-p3` branched from daemon-node master `64e191a`; same ledger/freeze/merge
discipline as P2; program ledger this file. Opus 4.8, git worktrees, P2 conventions.

### Lane G — CUDA engine arm

The `cuda` cargo feature exists ([daemon-train](../../crates/coprocessor/daemon-train)) and burn-cuda
runs a real op on the RunPod 4090, but the engine has **no `BackendKind::Cuda`** — the 4090 peer trains
on the CPU det lane today.

- Add the `BackendKind::Cuda` arm mirroring the wgpu arm (device init from probe, autotune integration,
  lazy host-boundary contract from B3).
- Run the full tolerance/parity/det-digest suites on the 4090 (`.#cuda-train` devshell + staged
  nvrtc-12.4 at `/root/cuda-rt-124`); CUDA joins the wgpu tolerance class, det lane stays host fp32
  (consensus unchanged).
- **Exit:** 160M single-host run on the 4090 with reference parity inside the Optimizer tolerance class
  + throughput record.

### Lane R — live checkpoint-resync

Engine API exists and is proven deterministic in-process (B4's `resync_by_replay` test); the live
worker rejoin path uses fresh-state.

- Wire §9 into the worker rejoin: coordinator exposes latest checkpoint pointer → rejoining worker
  fetches via payload store → `resume_from_checkpoint` → replay retained rounds → rejoin roster
  byte-identical.
- Small cloud half: checkpoint-pointer surface on the DO (additive; redeploy dev coordinator per the
  keep-in-sync rule). **This half lands early on the daemon-cloud coordination branch
  `swarm/p3-integration` so Lane S's runs can consume it.**
- **Exit:** churn drill upgraded — the e2e asserts the rejoiner's post-rejoin digests byte-match
  survivors (removing B4's documented exclusion in `fleet_live_hetero.rs`).

### Lane S — 160M fleet staging (artifact distribution)

The gate pre-staged wasm modules and used a synthetic corpus; the 160M envelope has never run across
the fleet.

- Module distribution: fetch experiment wasm by content hash from the payload store (presign plane),
  verify blake3 before instantiation — removing `DAEMON_TRAIN_MODULE` pre-staging.
- Corpus staging: pre-tokenized TinyStories shards published to R2; workers fetch assigned shards per
  the envelope `[data]` section.
- Stage the 160M module + corpus on all four fleet machines' caches; VRAM/RAM budgets already validated
  per-platform by the probes.
- **Exit:** the deferred measurement run — 160M across ≥4 heterogeneous peers on the real substrate,
  producing the direct ε-convergence figure and the <15% overhead measurement (replacing both gate
  caveats), with `--observe` + `swarm-replay` evidence.

### Merge order and dependencies

- Lanes **G and S are independent**; **R has a small cloud-half** that S's runs will consume — land R's
  DO change early via the coordination branch `swarm/p3-integration`.
- **Merge 1:** G + R integrated, churn-with-resync drill green on wrangler-dev.
- **Merge 2 (program gate):** the 160M fleet measurement run (S) with G's CUDA peer and R's resync
  active — the strongest end-to-end evidence the stack has produced.
- **Handoff:** ledgers, superproject gitlink proposal (human signature), updated spec §16 caveat
  removal (the §17 gate's two scale caveats).

### Sequencing note

The SigV4 presign flip (Phase-0 item 3) benefits Lane S's payload-heavy 160M runs — direct-to-R2
instead of proxying GBs through the Worker. It is a human R2-token step + an agent redeploy; the
mechanism is verified ready (see "SigV4 presign — readiness" below).

## Carried follow-ons from P2 relevant to these lanes

From `swarm-p2-ledger.md` "Carried follow-ons (the P3-and-beyond register)" — the items P3 consumes:

1. **CUDA engine arm (`BackendKind::Cuda`)** — **Lane G.** Deps + `.#cuda-train` devshell + staged
   nvrtc 12.4 (`DAEMON_CUDA_RUNTIME_DIR`, `/root/cuda-rt-124`) are **ready**; the worker still trains
   the CPU det lane on CUDA boxes. The `cuda` cargo feature (`burn/cuda`) is already in the merged
   trunk (cudarc runtime-dlopen, lock-neutral, no toolkit at build). The `.#cuda`/`.#cuda-train` flake
   stanza landed at P2 Wave-3 (adjudication (c)): unfree-scoped `cudaPackages_12_x.cuda_nvrtc` keyed to
   the box driver + `cuda_cudart` headers + `CUDA_PATH`/`LD_LIBRARY_PATH` wrapper; **build on the CUDA
   box** (RunPod 4090 / Windows 5090), one sealed `nix build` at most.
2. **Live checkpoint-resync in the worker rejoin** — **Lane R.** B4's design note: surface the latest
   `CheckpointManifest` (additive cloud pointer) → `resume_from_checkpoint` → replay retained rounds;
   upgrades the churn assertion from "run finishes" to "rejoiner byte-identical". A-lane + small cloud
   addition. Engine API proven deterministic in-process (B4 `resync_by_replay`); the live rejoin path
   currently uses fresh-state (per B4 the exclusion is documented in `fleet_live_hetero.rs`).
3. **`--observe` in the cloud-DO worker loop** (Merge-2 Task-5 note) — **Lane S evidence.** The observe
   surface currently rides the `swarm-local`/`live_transport` harness only; the `ws_live_workers` /
   `fleet_gate_ceremony_with_churn` worker-subprocess loop drives `TrainSupervisor`+`SwarmService`
   directly and does not yet wire `--observe`. Wiring it gives direct offline-replayable capture from a
   worker-subprocess gate run — the strongest S evidence.
5. **SigV4 R2 token** (Risk-5 checklist) — **Lane S payload plane.** The dev substrate rides the
   object-proxy plane; direct SigV4 presign to the real bucket is the production path (mechanism ready;
   human mints the token). Direct-to-R2 avoids proxying the 160M payloads through the Worker.
7. **160M-at-scale staging** — **Lane S.** Stage the 160M envelope + tokenized corpus on the fleet; run
   the overhead tool + capture the swarm loss curve at scale — **this is the program gate**, converting
   both §17 caveats (3, 4) into measurements.
9. **RunPod bare-env WS-dial posture** — **Lane S fleet staging.** The ceremony runs the pod worker
   fine after an on-box rebuild; artifact-drift on ephemeral pods is the real lesson (the worker
   fail-fast guard catches a `swarm-net`-less build loud). Consider a build-fingerprint print at worker
   startup. Relevant when staging the 160M module/corpus caches on the pod.

Other P2 carried items (4 sentry upstream, 6 workers.dev auth posture, 8 `late_join` drill de-flake,
10 P3 app-surface) are **out of scope for these three lanes** — noted for completeness, not P3-lane
work.

## Live substrate endpoints (P2, still current — do NOT re-provision)

Verified current at the P2 gate ceremony (2026-07-14); reuse as-is:

- **Coordinator:** `https://daemon-swarm-dev.me-dc6.workers.dev` — deployed `daemon-swarm-dev` worker
  (version `95cbb0f1`), matches daemon-api `swarm/p2-integration`@`b13f51d`. API base
  `/api/v1/swarm`; wss `…/runs/:id/ws`; object-proxy presign plane (SigV4 when R2_* secrets set); the
  `x-daemon-org-id`/`x-daemon-actor` internal-identity headers on workers.dev (no gateway; dev only).
  **KEEP-IN-SYNC rule:** any daemon-cloud coordination-branch change touching
  `coordinator-wasm`/`registry.ts`/`shell.ts` (e.g. Lane R's DO checkpoint-pointer) must be redeployed
  via `apps/swarm/scripts/deploy-dev.sh` (or a rendered `wrangler deploy`) — a stale deployment was a
  real P2 Merge-2 bug.
- **iroh relay:** `http://51.159.120.241:3340` (M1 mini) — `generate_204` → 204 at the gate.

### SigV4 presign — readiness (Phase-0 item 3; no changes needed, verified on `b13f51d`)

The production SigV4 plane is fully wired on daemon-api master; only the human R2-token + an agent
redeploy remain. When `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` are exported and
`apps/swarm/scripts/deploy-dev.sh` re-runs:

- deploy-dev.sh (lines 62–69) detects all three and pushes them as `wrangler secret put` on
  `daemon-swarm-dev`; absent → object-proxy plane (message printed).
- `presign.ts::hasSigV4Creds(env)` then returns true → `presignObject` routes to `sigV4PresignUrl`
  instead of `objectProxyUrl` — **the worker auto-switches planes at runtime, no code change**.
- SigV4 presigns against `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com/<bucket>/<key>` with
  `region=auto`, `service=s3`; `bucket = env.SWARM_R2_BUCKET = "swarm-dev"` (from `wrangler.dev.jsonc`
  vars), path-style; signature math in the KAT-tested `./sigv4` module. The R2 token must cover the
  **`swarm-dev`** bucket (Object Read & Write); account id `dc6bce79dcd681b757dcd2f24556b3e4`.
- `live-smoke.mjs` reports the plane by checking the presign URL host: prints
  **`presign plane: SigV4/real-R2`** when it contains `r2.cloudflarestorage.com`, then does a
  direct-to-R2 PUT → GET byte-identical round-trip. (The `{url, expires_at, headers}` contract is
  identical across planes — B1's `R2Store` never sees the difference.)

## P2 frozen seams the lanes MUST respect (extend additively only)

Everything frozen through the P2 exit gate is inherited. The seams these three lanes touch or build
against, verbatim from `swarm-p2-ledger.md` Merge-1/2/3 frozen-interface sections:

- **`tabi@1` (66 ops) — FROZEN FOREVER** at the P1 exit gate (spec §16). Additive `op@version` growth
  only; a breaking change is `tabi@2`. **Lane G adds NO `tabi` ops** — a `BackendKind::Cuda` arm is an
  engine backend, not an ABI change; the det lane stays host fp32 (consensus unchanged).
- **Wire v42 — FROZEN.** `WireVersion::CURRENT == 42`; `SwarmHardwareReport.shared_mb: u64`
  (`#[serde(default)]`); `contract_wire_version_is_v42`. No P3 lane bumps it (see the wire note above).
- **`JoinCredentials` canonical-CBOR contract (A3, Merge-2):**
  `JoinCredentials`/`WsAuthSpec`/`IrohCredentials`/`IrohRosterPeer`/`EngineParams` (verbatim in
  `swarm-ledger-p2-a3.md §2`). Lane R's rejoin path composes with this (`resolve_join` → `AssessRun` →
  `JoinRun` credentials; assess runs before the engine consumes `EngineParams`); it does **not** edit
  the contract.
- **Observe surface (B2, Merge-2):** `--observe <dir>` (`<run>.dsmlog`+`<run>.dsmcap`),
  `swarm-replay <dir>`, `SwarmRun::{message_log,run_capture,write_observe}`, `verify_observe_dir`,
  `daemon_swarm_observe::{RunCapture,replay_from_state,replay_capture,logged_round_records}`. Lane S's
  measurement run consumes this for evidence; wiring `--observe` into the worker-subprocess loop is
  additive (carried follow-on 3).
- **B3 lazy-backend host-boundary inventory (the residency contract, ABI §5.9 unchanged):** det lane,
  scalar/metric readouts, `canonical_state_bytes`, `checkpoint_bytes`, `upd_push_tensor`, `grad@1`
  fold, `MetaReport`. `OpBackend`/`TrainerBackend` traits unchanged. **Lane G's `BackendKind::Cuda` arm
  MUST honor this host-boundary contract** exactly as the wgpu arm does (lazy device-resident results;
  host copies only at the inventoried boundaries).
- **Declared-RunConfig (both halves, Merge-2):** `CreateRunRequest.{warmup_timeout_s,round_timeout_s,
  cooldown_s,global_batch,witness_target}` (additive optional) → `registry.ts` validate + verbatim
  `/init` forward → `ShellConfig` → `coordinator-wasm InitConfig` `#[serde(default)] Option`s
  (declared-over-default; the registry NEVER parses the envelope). Lane R's checkpoint-pointer surface
  is additive to the DO in the same spirit; Lane S authors the 160M create-request via
  `swarm-local --emit-create-request`.
- **Ceremony harness (Merge-3):** `fleet_gate_ceremony_with_churn` + env knobs
  (`SWARM_GATE_DROP_INDEX`/`SWARM_GATE_DROP_AFTER_ROUND`); `fleet_live_hetero.rs` env contract
  (`SWARM_FLEET_*`); the worker's loud `swarm-net`-less-live-attach `Error` (behavioral contract: live
  credentials + no feature = Error, never silent fallback); `swarm-p2-gate-runbook.md`. Lane R removes
  B4's documented resync exclusion in `fleet_live_hetero.rs`; Lane S drives the 160M run through this
  harness.
- **Tier-1 gate:** `cargo run -p xtask -- swarm-ci-det` (guests + pinned CPU consensus suites). Keep it
  green on the trunk after every merge.
- **Guest guard:** warn-and-rebuild manifest guard; canonical trunk manifest carried from P2
  (`test_abi_basic e2a8780e…`, `tiny_llama 3bf68973…`). The guest bytes are keyed on the absolute
  checkout path (Merge-1 adjudication) — regenerate canonically on each worktree; `ensure_built()`
  always rebuilds before load, so the module in use is fresh.
- **FROZEN files (single-writer, integration-owner only):** root **`Cargo.toml`** (workspace members
  glob, `[workspace.dependencies]`, `[workspace.lints]`, profiles), **`deny.toml`**, **`flake.nix`**
  (devShell/package lanes incl. `.#cuda-train`). New third-party deps / features-that-pull-new-crates
  route through the integration owner (who re-runs `cargo deny`). Adding a new member crate is fine
  (glob picks it up); adding a *feature of an already-declared workspace dep* from your own crate's
  manifest is lane-owned.

## Conventions inherited from P1+P2 (carry over verbatim unless noted)

- **Worktree ownership** — one lane, one worktree; disjoint file ownership is the merge guarantee.
- **Commit styles:** `feat(...)`/`fix(...)`/`build(workspace|deps|nix)/(...)` per change; **lane
  ledgers** land as `mirror(<lane>): ...`; **integration/merge** records + this program ledger land as
  `mirror(<wave|merge>-N): ...`. Merges are `--no-ff` (ort); disjoint ownership keeps `Cargo.lock` the
  only co-touched file (git auto-merges the additive regions), plus `guests/guests.blake3` when a lane
  recompiles the guest `.wasm` (reconcile by regenerating canonically on the trunk).
- **daemon-node does NOT sign** (`commit.gpgsign=false` by submodule convention). The **superproject**
  requires GPG-signed commits with explicit human approval — every superproject change is a *proposal
  for the human*, never committed by an agent (justfile lint fix, spec §16/§17 caveat removal, gitlink
  bump).
- **`build-guests` after every checkout (P1's hardest-won lesson):** the wasm guests live under the
  gitignored `guests/target/**` and do NOT travel with a branch. **ALWAYS run
  `cargo run -p xtask -- build-guests` in a fresh worktree before any wasm-backed test.**
- **Resource discipline (non-negotiable):** cap every build at ≤ nproc/2 (`CARGO_BUILD_JOBS`,
  `-j N`, `nix build --max-jobs 1 --cores N`); **one build at a time**; never `just lint-all` /
  whole-tree clippy sweeps; verify kills with `pgrep -f 'cc1plus|rustc|makensis|ninja'`. Keep cargo's
  `CARGO_TARGET_DIR` repo-local via the devShell.
- **Billing-stall note (carried):** long nix builds and multi-minute `cargo test`/`nix build` runs
  produce no output for stretches (workspace test ~6 min; release wgpu 160M parity ~4–6 min cold; a
  full CUDA build on-box is heavier). A silent long build is **expected, not hung** — background with a
  completion sentinel, do not stack a second build against it. **Self-heals in minutes** for transient
  substrate/ssh stalls.
- **Disk hygiene:** on lane completion, reclaim the finished lane's build artifacts with
  `/home/j/experiments/daemon-worktree/clean-lane-target.sh <worktree-dir>` (deletes ONLY `target/`
  and `guests/target/`; refuses on uncommitted tracked changes or an active build). Prune lane targets
  between waves to keep `$HOME` off the swap cliff.
- **Known flake — never modify:** the `daemon-conformance` detached-delegation/operator-steer trio +
  `drills.rs::late_join_mid_run_syncs_and_contributes` are nondeterministic under full parallel load;
  **pass-in-isolation = green** is the standing disposition. No swarm lane touches `daemon-conformance`.

## Fleet inventory (P2-provisioned; reuse — see `daemon-worktree/fleet-report-p2.md`)

Totals: **5 peers, 3 GPU vendors (AMD, Apple, NVIDIA), 3 OSes (Linux, macOS, Windows)**, 4 distinct
WAN network locations (M1+M4 are co-located ~1 ms — treat as one location for the "distinct networks"
check). SSH targets resolved by the P2 fleet-provisioning pass:

| Peer | SSH target | GPU / backend | Memory | P3 lane relevance |
|---|---|---|---|---|
| **Strix Halo** (this box) | local | AMD Ryzen AI Max+ 395, RADV/Vulkan (gfx1151), UMA | 128 GB UMA | trunk/dev box; RADV `.#vulkan` reference peer; **hosts the iroh relay :3340** (bring up in-wave) |
| **M1 Mac mini** | `m1@51.159.120.241` | Apple Metal (M1) | — | Metal peer; **iroh relay host** `http://51.159.120.241:3340` |
| **M4 Mac** | `m1@62.210.193.129` | Apple Metal (M4) | 32 GB | Metal peer; flake eval needs `bubblewrap` Linux-gating (P2 C-lane note) for on-box devShell |
| **RunPod RTX 4090** | `ssh -p 13988 root@213.173.109.230` | NVIDIA CUDA (Ada), driver 550.127.05 | 124 GB RAM, `/` 100 G local, `/workspace` shared netfs | **Lane G CUDA target** (`.#cuda-train`, nvrtc 12.4 `/root/cuda-rt-124`); repo on local `/root/daemon-node`; no systemd (docker-init); ephemeral — watch artifact-drift (follow-on 9) |
| **Windows Server 2022 + RTX 5090** | `usergpu356@37.230.134.194` | NVIDIA CUDA + Vulkan 1.4.341 (5090, 32 GB), driver 610.74 | — | Windows CUDA/Vulkan peer; **deploy MinGW cross-built worker, never build on-box**; worker `.exe` link blocked by the sentry/MinGW minidump issue (P2 follow-on 4/f) — decide probe-only vs feature-gate |

Fleet caches to stage for Lane S: the 160M module + tokenized TinyStories corpus shards on all four
run peers (Strix/RunPod/Windows + one Mac), budgets already probe-validated per-platform.

---

## Wave-0 scaffold record

Landed on `integrations/swarm-p3` (base `64e191a`). This `mirror(p3-wave0): program ledger` commit is
the first (and only) Wave-0 trunk commit.

### Trunk / worktrees / branches created (this wave)

daemon-node (all from base `64e191a`):

| Role | Branch | Location | Base |
|---|---|---|---|
| **Trunk (integration owner)** | `integrations/swarm-p3` | `/home/j/experiments/daemon-worktree/swarm-p3-integration` | `64e191a` |
| **Lane G** (CUDA engine arm) | `swarm/g` | `/home/j/experiments/daemon-worktree/p3-g` | `64e191a` |
| **Lane R** (live checkpoint-resync) | `swarm/r` | `/home/j/experiments/daemon-worktree/p3-r` | `64e191a` |
| **Lane S** (160M fleet staging) | `swarm/s` | `/home/j/experiments/daemon-worktree/p3-s` | `64e191a` |

daemon-cloud `daemon-api`:

| Role | Branch | Location | Base |
|---|---|---|---|
| **Cloud coordination** (Lane R DO half) | `swarm/p3-integration` | `/home/j/experiments/daemon-cloud/daemon-api` (branch only, no worktree) | master `b13f51d` |

The daemon-cloud coordination branch carries no changes yet — Lane R's DO checkpoint-pointer half lands
there (additive), then the dev coordinator is redeployed per the KEEP-IN-SYNC rule. daemon-cloud is
**not gitlinked** to this trunk; the runtime contract is the presign/WS HTTP surface + (new for R) the
checkpoint-pointer surface.

### Wave-1 launch notes (G / R / S)

- **Lane G:** build the `BackendKind::Cuda` arm mirroring the wgpu arm; honor the B3 lazy host-boundary
  contract; det lane stays host fp32. Full tolerance/parity/det-digest suites on the RunPod 4090 via
  `.#cuda-train` (nvrtc 12.4 at `/root/cuda-rt-124`); **build on the CUDA box**, one sealed `nix build`.
  No `tabi` ops, no wire change. `build-guests` after checkout.
- **Lane R:** wire the coordinator checkpoint-pointer → payload-store fetch → `resume_from_checkpoint`
  → replay retained rounds into the live worker rejoin; land the small DO checkpoint-pointer surface on
  `swarm/p3-integration` **early** (additive; redeploy dev coordinator). Compose with the frozen
  `JoinCredentials` path (do not edit it). Exit: remove B4's resync exclusion in `fleet_live_hetero.rs`
  and assert rejoiner post-rejoin digests byte-match survivors.
- **Lane S:** content-hash module fetch from the payload store (blake3-verify before instantiate,
  remove `DAEMON_TRAIN_MODULE` pre-staging); pre-tokenized TinyStories shards to R2 per envelope
  `[data]`; stage the 160M module+corpus on the fleet caches. Consumes the SigV4 plane (direct-to-R2)
  and R's checkpoint-pointer. Exit = the program gate: 160M across ≥4 heterogeneous peers, direct ε +
  <15% overhead measurements, `--observe`+`swarm-replay` evidence.
- **All lanes:** frozen files locked (route new deps/features through the integration owner); run
  `build-guests` after every checkout/rebase; cap builds at ≤ nproc/2, one at a time.

### Gate results (Wave-0, trunk HEAD = this ledger commit on `64e191a`)

Sanity-only (it is master — expected green):

- `nix develop --command cargo fmt --check` — recorded below.
- `cargo run -p xtask -- swarm-ci-det` (tier-1 gate: builds guests + pinned CPU consensus suites) —
  recorded below.

---

## Merge-1 integration record (G + R)

Merge-1 lands the two engine follow-ons — **Lane G** (CUDA engine arm) and **Lane R** (live
checkpoint-resync) — onto the shared trunk. Lane S (160M fleet staging) is **NOT** in this merge; it
lands at Merge-2 (the program gate). Integration owner: Merge-1 owner. Verified 2026-07-14 on the
Strix Halo (AMD RADV) dev box + the RunPod 4090 evidence carried from Lane G + the deployed dev
coordinator.

### HEADs

| Repo | Branch | HEAD | Note |
|---|---|---|---|
| daemon-node | `integrations/swarm-p3` | see `git log` (this ledger commit) | trunk after both `--no-ff` merges + the flake follow-on |
| daemon-node | `swarm/g` | `011dd4f` | merged (`mirror(G): finalize ledger — Merge-1 seams + 4090 results + P3 throughput record`) |
| daemon-node | `swarm/r` | `afdc129` | merged (`mirror(R): ledger — results`) |
| daemon-node | `swarm/s` | `3afec18` | **NOT merged** — lands at Merge-2 (Lane S still running; worktree/branch untouched) |
| daemon-cloud `daemon-api` | `swarm/p3-integration` | `6b0d978` | R's cloud half; **deployed** to the dev coordinator as version `8ca3579a`; `master` untouched; nothing pushed |

### Merge commits + conflicts

- `Merge branch 'swarm/g' into integrations/swarm-p3` (ort, `--no-ff`) — clean; 15 files (the
  daemon-train engine/backend/autotune arm + cuda suites + the G ledger + throughput doc + the
  one-line `Cargo.lock` cudarc edge).
- `Merge branch 'swarm/r' into integrations/swarm-p3` (ort, `--no-ff`) — 12 files;
  **`daemon-train-worker/live.rs` auto-merged** (the only co-touched engine file). G inserts
  `worker_engine_config()`/`select_backend()` wiring at the `build_wasm_backend` + `LadderBackend`
  call sites; R inserts the resync path (`resync_on_join`, `build_registry`, `WorkerStore::committed_set`,
  the `Checkpointed`-publish forwarder, the `EngineEvent::Resynced` arm). The two lanes touch
  **disjoint base-line regions** of `live.rs`, so ort composed them with no textual conflict — verified
  post-merge: both `build_wasm_backend` call sites route through `worker_engine_config()` → `select_backend()`,
  and the rejoin/resync path is intact.
- **`Cargo.lock`:** G's single additive `cudarc 0.19.8` edge line only; R added no deps → no lock
  conflict. **Guests:** neither lane changed a guest `.wasm` source (both list `guests/` off-limits);
  the committed canonical manifest (`test_abi_basic e2a8780e…`, `tiny_llama 3bf68973…`) is identical
  on both lane HEADs, so no manifest conflict. `build-guests` re-run on the trunk; the resulting
  per-worktree path-keyed drift (`d370038…`/`d9aa630…` in this checkout) is **NOT committed** (the
  standing path-keyed-codegen rule; `ensure_built()` rebuilds before load so the module in use is
  fresh — the guard's warn-and-rebuild is the mechanism, seen firing green in the runs below).

### Hard gates (the two user gates, on the MERGED trunk, this AMD box)

1. **Fat-worker graceful degradation on non-NVIDIA — PASS.**
   `cargo test -p daemon-train --features cuda --test cuda_lifecycle cuda_probe_degrades_cleanly_without_nvidia`
   green on this AMD box: `probe_cuda()` → clean `None` (missing-libcuda dlopen unwind caught,
   memoized-consistent), `cuda_adapter_available() == false`, `cuda_nvrtc_ready() == false`, **no
   panic/abort**, and the CPU det lane still constructs + `da_build`s. Re-ran on the merged trunk (G
   originally proved it on its lane HEAD).
2. **No link-mode cudarc in the merged graph — PASS.**
   - `cargo tree -p daemon-train -e normal` → **no** cudarc / cubecl-cuda / burn-cuda (default graph
     clean; the `cuda` feature is off-default).
   - `cargo tree -p daemon-train --features cuda` → cudarc `0.19.8` activated features =
     `cuda-version-from-build-system, driver, fallback-dynamic-loading, fallback-latest, nccl,
     nccl-02030, nvrtc, std` — **no `dynamic-linking` / `static-linking`** feature anywhere (dlopen
     mode).
   - ELF `readelf -d` on the built fat worker (`daemon-train-worker --features swarm-net,cuda`):
     `DT_NEEDED` = `libgcc_s.so.1`, `libm.so.6`, `libc.so.6`, `ld-linux-x86-64.so.2` — **no
     `libcuda`/`libnvrtc`/`libcudart`/`libnvidia-*`**. Confirmed no link-time CUDA dependency; the
     driver/NVRTC symbols go through lazy `libloading` dlopen.

### Adjudications (verdicts)

1. **Lane G deviation 2 — the wgpu rung of `select_backend()` now activates wgpu in LIVE workers
   (previously CPU-only). VERDICT: ADOPT (keep enabled). Safe for the Merge-2 fleet run.**
   - **Probe-gated:** the wgpu rung fires only when the `wgpu` feature is built **and** the memoized
     `catch_unwind`-wrapped `probe_wgpu()` reports a usable adapter; otherwise it falls to the CPU det
     lane. `DAEMON_TRAIN_BACKEND=cpu` is the operator escape hatch.
   - **Consensus-invariant:** the det lane materializes host fp32 (`self.host` + `det_core`, shared
     with `CpuBackend`) for every `det_*` op, so `canonical_state_bytes` (the consensus digest) is
     backend-independent. Only the tolerance-class native lane (forward/backward/AdamW) runs on the
     GPU. **Verified on `.#vulkan` (RADV):** `wasm_backend_determinism::cross_backend_wgpu::*`
     (sparse_loco / diloco / demo, 6 rounds each) — **cpu-vs-wgpu det digests byte-identical every
     round**, native payloads diverge (tolerance-class), both losses fall. This is the exact
     consensus tripwire and it is green on the merged trunk.
   - **Why ADOPT, not just tolerate:** at 160M the CPU native lane is intractably slow on the
     Metal/RADV fleet peers; enabling the probe-selected GPU native lane is what makes the Merge-2
     fleet measurement run tractable end-to-end, while the det/consensus bar is provably unchanged.
     This is the intended shape of the program gate (real GPU peers: CUDA on the 4090, wgpu on
     RADV/Metal). Recorded as a frozen Merge-1 seam (below).
2. **Lane G deviation 1 — `worker_protocol` GPU-count assertion widened (`gpus == 0` →
   `gpus <= 1` for GPU-featured builds). VERDICT: ACCEPT.** The prior assertion was factually
   outdated by the cuda arm (a cuda/wgpu build legitimately reports one GPU); one-line,
   behavior-preserving for every existing build shape; `worker_protocol` green on `.#vulkan` (4
   tests) and in the cuda suite.
3. **Lane R finding — checkpoint objects are per-peer byte-divergent (local Adam optimizer state is
   in `checkpoint_save`). VERDICT: ACCEPT as-is for Merge-1; record the follow-on.** Byte-identity of
   the *rejoiner* holds today because the resync loads whichever valid post-`round` checkpoint object
   is stored (verified by its own blake3) and replays — the digest + replay depend only on the
   **consensus half** (params + replicated persistents). The DO pointer's `cross_checked` therefore
   means "≥2 peers uploaded a manifest", **not** "byte-identical checkpoint bytes" (the live drills
   below show `cross_checked:false, uploads:1` yet byte-identical rejoin). **Follow-on (Lane G/B
   territory, §9, NOT Merge-1):** if a future lane wants `cross_checked` to imply byte-identical
   checkpoint bytes, `checkpoint_save` must exclude per-peer local state — a daemon-train backend
   change, out of R's scope.

### Cross-lane composition evidence (on the MERGED trunk)

- **R's resync drill WITH G's `select_backend` present** — `checkpoint_resync.rs` (merged-trunk
  worker built `--features swarm-net`, so `select_backend()` is compiled + exercised, resolving to
  `Cpu` on this box) run against the **deployed `8ca3579a` coordinator**: `resync_rejoiner_is_byte_identical`
  + `fresh_state_rejoin_still_finishes` both GREEN. Rejoiner resynced from checkpoint round 5;
  post-resync rounds 6 & 7 digests byte-identical across all 3 peers incl. the rejoiner. (This run
  doubles as the trunk↔cloud coherence check — see below.)
- **Upgraded ceremony harness** — `fleet_gate_ceremony_with_churn` compiles (`--features iroh`) and
  its **local variant is GREEN** with 3 local peers / 8 rounds against the deployed coordinator: drop
  peer 2 after round 2 → coordinator parks (floor breach at round 6) → checkpoint-resync rejoin →
  rounds 6 & 7 byte-identical to survivors, run Finished, B4's rejoiner-digest exclusion removed and
  the rejoiner IS in the byte-identity assertion. **Harness note:** the *default* 2-LOCAL-peer
  self-check does NOT drive the drop→park→rejoin cycle (a lone survivor finishes the run before the
  floor breaches), so the churn ceremony needs **≥3 peers** — a non-issue for the Merge-2 fleet run
  (≥4 peers) but recorded so future local runs configure `SWARM_FLEET_PEERS` with ≥3.
- **Full parity/digest suites on `.#vulkan`** — `cargo test -p daemon-train --features wgpu` GREEN:
  lib 43, `burn_wgpu_parity` 18, `wasm_backend_determinism` 12 (incl. `cross_backend_wgpu`),
  `wgpu_lifecycle` 3, `worker_protocol` 4, `guest_lifecycle` 9, `preset_160m` 2, `abi_surface` 2. The
  `#[ignore]`'d 160M-release wgpu parity cases (`reference_parity_wgpu` ×3, `preset_160m_wgpu` ×1) are
  the heavy Merge-2 fleet-ceremony runs, deferred by design (P2 already holds the RADV 160M record).
- **Fat-union clippy** — `-p daemon-train --features swarm-net,cuda,wgpu -D warnings` GREEN (plus
  `cuda`, `swarm-net,cuda`), in `.#cuda-train`.

### Trunk↔cloud coherence (optional user task — DONE)

R's live resync drill (merged-trunk worker) ran GREEN against the already-deployed dev coordinator
`8ca3579a` (R's cloud half `swarm/p3-integration`@`6b0d978`). The coordinator's `/state` carries the
additive `checkpoint` pointer, the checkpointer-publish POST registers it, and the rejoiner reads it
+ resyncs byte-identically — proving the merged node trunk and the deployed cloud half are coherent.
(No cloud redeploy in Merge-1: the deployed `8ca3579a` already matches the coordination branch tip.)

### Packaging design decisions (formalized — user decisions, recorded in the lane ledgers this wave)

- **D5 — ONE fat worker binary.** Backend packaging is a single worker with `ndarray + wgpu + cuda`
  features unioned (cuda target-gated to linux/windows x86_64 at packaging time), the **runtime probe
  ladder** (`select_backend()`: **CUDA → wgpu → CPU**) selecting the arm. HARD GATE (proven above):
  a cuda-featured worker on non-NVIDIA degrades cleanly — no panic, **no link-time CUDA dependency**
  (dlopen mode; ELF `DT_NEEDED` clean).
- **D6 — nvrtc = fetch-on-demand (asset-style), driver-keyed.** NVRTC is fetched on demand keyed by
  the detected driver version, staged into `DAEMON_CUDA_RUNTIME_DIR`; until staged, the probe
  downgrades to wgpu/CPU (two-leg readiness gate: a **loadable `libnvrtc`** AND the **complete cudart
  JIT include tree** at `CUDA_PATH`). Building the fetcher is a later item; the **contract** is frozen
  here (see "Frozen Merge-1 seams"). The launcher must export **both** `DAEMON_CUDA_RUNTIME_DIR` (→
  `LD_LIBRARY_PATH`) and `CUDA_PATH` (→ the driver-matched include tree). Merge-1 codifies the
  `CUDA_PATH` export into the `.#cuda-train` shellHook (below).
- **No-Nix-at-runtime for the shipped base distribution (user constraint, recorded).** The shipped
  base distribution MUST NOT depend on Nix at runtime; the dlopen targets are the **end-user system's
  standard paths** — Windows `System32\nvcuda.dll`, Linux ldconfig `libcuda.so.1`. (The `.#cuda-train`
  devShell + `DAEMON_CUDA_RUNTIME_DIR` are the *developer/CI* staging mechanism, not the shipped
  runtime contract.)
- **Static-linkage wrinkle (packaging follow-on, NOT this merge).** dlopen from a fully-static binary
  is the open question for the bundle work: the fat worker's final linkage shape (fully-static vs
  dynamic libc) must be verified to still permit `libloading` dlopen of the system driver when
  packaging lands. Recorded as a bundle-work follow-on; worker linkage shape to be verified then.

### Flake follow-on applied (integration-owner right; documented)

Applied G's one-line `.#cuda-train` shellHook follow-on: when `DAEMON_CUDA_RUNTIME_DIR` is set, also
`export CUDA_PATH="$DAEMON_CUDA_RUNTIME_DIR"` (so cubecl-cuda's NVRTC kernel JIT resolves
`#include <cuda_runtime.h>` against the **driver-matched** staged include tree, not the build-time
nix `cudatoolkit` — a different CUDA level; G's 4090 live-attach smoke proved this must match).
**Trivially safe:** guarded on the var being set, so it is a **no-op on this AMD box** (var unset →
`CUDA_PATH` stays the nix `cuda-merged-12.9`, the note prints) — verified by re-entering the shell
both ways (unset → nix path; set → the runtime dir, with `LD_LIBRARY_PATH` head matching). `nix
develop` re-eval is instant (no derivation/package change). This is the daemon-node submodule
`flake.nix` (no signing); FROZEN-file edit made by the integration owner per the frozen-file rule.

### Frozen Merge-1 seams (extend additively only)

Everything the P1+P2 frozen sections list remains authoritative. Merge-1 additionally freezes (all
additive; `tabi@1` untouched, **wire stays v42**):

- **Lane G — the backend selection ladder + CUDA engine arm:**
  - `daemon_train::BackendKind::Cuda` (feature `cuda`) + `BurnCudaBackend = BurnBackend<Autodiff<Cuda>>`
    + `cuda_adapter_available()`. A type-parameter swap behind the unchanged generic `BurnBackend<B>`;
    `OpBackend`/`TrainerBackend`/`EngineConfig`/`WasmBackend` shapes unchanged (rides the existing
    `EngineConfig.backend` + `gpu_index`). Honors the B3 §5.9 lazy host-boundary residency contract by
    construction (shared generic).
  - `daemon_train::autotune::{CudaProbe, probe_cuda, cuda_nvrtc_ready, cuda_device_limits}` (feature
    `cuda`). **DeviceLimits CUDA source:** discrete — `vram_mb` from the driver `cuDeviceTotalMem`
    (24210 MiB on the 4090 container), `shared_mb = 0`, `unified = false`, `max_alloc_mb = total VRAM`
    (no per-buffer driver ceiling). Distinct from the wgpu/sysfs and Windows/macOS FFI sources.
  - The worker `select_backend()` **probe order CUDA → wgpu → CPU** (feature-gated, `swarm-net`-only;
    each rung requires its feature built + a usable probe, cuda additionally requires the D6 NVRTC
    readiness gate) + the `DAEMON_TRAIN_BACKEND=cpu` escape hatch. The wgpu rung is **live** (adopted
    above). `hardware()`/`device_limits()` report the cuda lane before wgpu when a device is present.
  - **The nvrtc fetch-on-demand contract (D6):** the runtime dir named by `DAEMON_CUDA_RUNTIME_DIR`
    must carry a driver-matched `libnvrtc.so.12` (+ `libnvrtc-builtins`), a resolvable `libstdc++`,
    and the **complete** cudart include tree (~186 entries incl. `crt/`, `cooperative_groups/`,
    `cuda/std/`); readiness = `cuda_nvrtc_ready()` flipping true in a fresh process (no sentinels).
    Base distribution dlopens the box driver's own userspace from standard system paths (no Nix at
    runtime).
- **Lane R — the checkpoint-pointer surface + resync ladder:**
  - **Cloud (DO) checkpoint-pointer surface** (`daemon-api` `swarm/p3-integration`, deployed
    `8ca3579a`): `POST /api/v1/swarm/runs/:id/checkpoint {round, hash, size}` (internal-auth) →
    `registerCheckpoint` fold → DO storage `checkpoint_pointer`; `GET /runs/:id/state` →
    `data.checkpoint = {round, hash, size, cross_checked, uploads} | null`. `coordinator.wasm`
    byte-unchanged (no `tick`/consensus change).
  - **Node resync ladder** (`daemon-train-worker` live attach, `resync_on_join`): GET `/state` →
    decide via `plan_resync` — no pointer / first epoch → fresh-state; `target == ckpt.round` →
    `resume_from_checkpoint` (HEAD the stored object for its own hash, blake3-verified load); gap ≤
    retention → `resync_from_checkpoint` (replay `record-set.cbor` + payloads via
    `R2Store::fetch_record_set_object`, in record order); gap > retention (`WaitForEpoch`) or any
    fetch miss → fresh-state + `Warning{class="resync"}`. Best-effort, never hard-fails the rejoin
    (a real fold fault surfaces `Event::Error{Desync}`).
  - **Additive proto/engine surfaces:** `EngineParams.payload_retention_rounds: u64`
    (`#[serde(default)]`; `0` = unbounded) — the §9 resync-replay window; `Event::ResyncProgress
    {round, from_checkpoint, replayed, total}` — additive off-wire telemetry via the A3 pump;
    `CheckpointManifest.size`; `RoundEngine::resync_from_checkpoint` + the `on_round_record`
    `<= last_ingested` resync-composability guard; net `RegistryClient::{fetch_state,
    publish_checkpoint}` + `CheckpointPointer`/`RunState`. `JoinCredentials`/`EngineParams` back-compat
    preserved (a pre-P3 buffer still decodes).

### Gate matrix (MERGED trunk — all green unless noted)

Run via `nix develop --command …`; builds capped at `CARGO_BUILD_JOBS=16` (nproc/2 = 16); one build
at a time.

- `cargo fmt --all --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- Feature combos `-D warnings` ✓: (default shell) `daemon-swarm-net ws,iroh` · `daemon-swarm-run
  iroh` · `daemon-swarm-e2e iroh` · `daemon-train swarm-net` · `daemon-train wgpu` · `daemon-train
  swarm-net,wgpu`; (`.#cuda-train`) `daemon-train cuda` · `daemon-train swarm-net,cuda` ·
  **`daemon-train swarm-net,cuda,wgpu`** (the fat-worker union).
- `cargo deny check` ✓ (advisories/bans/licenses/sources — cudarc adds no crate).
- `cargo test --workspace --no-fail-fast` — **3327 passed, 2 failed**; the 2 failures are the standing
  known flake `daemon-conformance::node::detached_delegation` under full parallel load — **green in
  isolation** (`-p daemon-conformance --lib node::detached_delegation --test-threads=1` → 5/5 pass).
  No swarm-lane file involved; the green-in-isolation disposition applies.
- `.#vulkan` suites ✓ (full `daemon-train --features wgpu` suite; the cross-backend det digest
  adjudication evidence above).
- `live_transport` (`daemon-swarm-e2e --features iroh`) ✓ — 7/7 (incl. `live_late_join_resyncs_over_iroh`,
  `live_flagship_three_peers_ten_rounds_all_agree`, `live_stall_ladder_recovers_over_iroh`).
- wasm32 `daemon-swarm-proto` + `daemon-swarm-coordinator` ✓.
- `cargo run -p xtask -- build-guests` ✓ (path-keyed drift not committed).
- `typos docs/specs` ✓.
- `cargo run -p xtask -- swarm-ci-det` (tier-1 CPU consensus gate) ✓.

Known flakes (green-in-isolation rule, never modified): the `daemon-conformance` detached-delegation
trio (observed + verified green in isolation this session), `f1_approvals_pending_is_owner_scoped`,
`drills.rs::late_join_mid_run_syncs_and_contributes`.

## Merge-2 launch notes (the program gate)

Merge-2 = the 160M fleet measurement run (Lane S) with G's CUDA peer + R's resync active — the
strongest end-to-end evidence the stack produces; converts spec §16/§17's two scale caveats
(ε-convergence, <15% round overhead) into direct measurements.

- **Lane S lands at Merge-2.** `swarm/s` @ `3afec18` was untouched by Merge-1 (still running in
  `/home/j/experiments/daemon-worktree/p3-s`). Merge-2 merges `swarm/s` into `integrations/swarm-p3`
  on top of this Merge-1 trunk; expect co-touched `Cargo.lock` (additive) + `live.rs`/worker-bin
  edges (S's content-hash module fetch vs the merged select_backend + resync path — reconcile as
  Merge-1 did) + `guests/guests.blake3` (regenerate canonically).
- **The 160M fleet measurement ceremony** runs through `fleet_gate_ceremony_with_churn` /
  `fleet_live_hetero.rs` with `SWARM_FLEET_PEERS` = the ≥4-peer heterogeneous fleet (Strix RADV /
  RunPod 4090 CUDA / Windows 5090 / one Mac Metal), `SWARM_FLEET_WS_URL` = the dev coordinator,
  `--observe` + `swarm-replay` for evidence. G's live GPU rungs (CUDA + the adopted wgpu rung) mean
  each peer trains its native lane on its GPU while the det/consensus lane stays host fp32
  (byte-identical). Configure ≥3 peers for any churn variant (the 2-peer self-check does not park).
- **Checkpoint-retention requirement Lane S MUST honor:** R's resync reads the retained
  `record-set.cbor` + round payloads + the `CHECKPOINT_PEER` checkpoint object (all existing §11.3
  keys). Lane S's R2 lifecycle rule **must retain the checkpoint objects
  (`runs/<run>/rounds/<round>/<cc..>.upd`) and record-sets ≥ `payload_retention_rounds`** — do not
  expire them earlier than the round payloads, or a fleet-scale rejoiner falls back to fresh-state.
  The pointer is queryable at `GET /state.checkpoint` for Lane S run tooling.
- **Cloud:** the deployed `8ca3579a` coordinator already carries R's checkpoint-pointer surface + the
  SigV4 presign plane (R2_* secrets set). If Merge-2 touches `coordinator-wasm`/`registry.ts`/`shell.ts`,
  redeploy via `deploy-dev.sh` per the KEEP-IN-SYNC rule (a stale deployment was a real P2 bug).
- **Handoff after Merge-2:** ledgers, the superproject gitlink bump (human GPG signature), and the
  spec §16/§17 caveat removal — none of which are agent-committed on the superproject.
