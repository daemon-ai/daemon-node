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
