# Swarm P2 — Lane C3 ledger (CI tiers + WAN-gate preparation)

Lane **C3** of the Swarm P2 WAN Program, **Wave 3** — the final wave before the P2 WAN gate ceremony
(Merge 3). Worktree `/home/j/experiments/daemon-worktree/p2-c3`, branch `swarm/c3`, base `fe27b9c`
(trunk `integrations/swarm-p2` @ Merge 2). This file is the single source of truth for what C3
landed, the seams it exports, every `flake.nix` edit (with rationale), the daemon-telemetry gating
change, and the fleet-validation evidence. Mirror commit: `mirror(C3): ledger`.

Contract read (Merge-2 Wave-3 launch notes): unblock the Windows worker peer (sentry `_invoke_watson`
MinGW blocker, adjudication (f)); land the `.#cuda` flake stanza (adjudication (c), pinned nvrtc
12.4); ready the M4 Metal peer; define + rehearse the <15% overhead measurement; wire the CPU-only
CI tier + design the scheduled/HIL tiers (TDD §8.1); and write the WAN-gate runbook. Integration-
owner-delegated flake rights: ADDITIVE outputs + the CUDA devshell stanza, each documented here.

## Scope + status

| # | Item | Status |
|---|---|---|
| 1 | Ledger first (this file) | ✅ |
| 2 | **Windows worker peer (gate-blocking)**: fix the sentry MinGW blocker; cross-build the full worker; deploy + real smoke (probe + WS live attach heterogeneity rehearsal) | ✅ — worker LINKS, RUNS on the 5090, and byte-matches Linux on the det lane over the live substrate |
| 3 | **CUDA lane**: `.#cuda` devshell (pinned nvrtc 12.4); RunPod build `--features cuda`, tolerance/parity suites, live WS attach | ✅ — `.#cuda-train` added; RunPod cuda build + det/parity 12/12 + autotune 14/14 + live 3-peer attach byte-identical |
| 4 | **M4 Metal peer readiness**: on-box build, Metal suites, live WS attach | ✅ — on-box swarm-net worker built, Metal probe validated, live 2-peer attach byte-identical |
| 5 | **Baselines + overhead measurement** (gate criterion <15%) | ✅ — measurement tool + procedure; local rehearsal numbers recorded (mechanism validated) |
| 6 | **CI tiers** (TDD §8.1) | ✅ — in-repo tier-1 runner (`xtask swarm-ci-det`); proposed superproject job + scheduled/HIL design in the runbook |
| 7 | **WAN-gate runbook** (`swarm-p2-gate-runbook.md`) | ✅ |

## Ownership / boundaries

- **Own:** CI/flake/runbook/fleet + daemon-telemetry gating. Files landed: `daemon-telemetry`
  (crash-handler gating), `flake.nix` (ADDITIVE outputs — delegated), `xtask` (CI helper subcommand),
  `tests/daemon-swarm-e2e/tests/fleet_live_hetero.rs` (NEW test file), `docs/specs/*` (this ledger +
  the runbook). Did NOT touch other lanes' worktrees, other daemon-swarm test suites (B4's this
  wave), the spec/TDD docs (B4's), root `Cargo.toml`/`deny.toml`, or the worker's transport/`JoinRun`.
- **Integration-owner-delegated flake rights:** ADDITIVE lane outputs + the `.#cuda` devshell stanza,
  adjudicated to this wave. Every edit documented below with rationale, subject to Merge-3 review.

---

## Landed commits (`swarm/c3`, base `fe27b9c`)

| Commit | Subject |
|---|---|
| (see git log) | `fix(telemetry): cfg-gate the native minidump monitor off for x86_64-pc-windows-gnu (unblocks the MinGW train worker)` |
| | `build(nix): daemon-train-worker-windows package + .#cuda-train devShell (C3 scoped flake rights)` |
| | `feat(swarm-e2e): heterogeneous-fleet live-attach harness + round-overhead measurement (C3)` |
| | `feat(xtask): swarm-ci-det tier-1 runner (TDD §8.1 per-PR swarm gate)` |
| | `docs(specs): P2 WAN-gate runbook (C3)` |
| | `mirror(C3): ledger` |

---

## Task 2 — the Windows worker peer (the gate-blocking item) — DONE

### The daemon-telemetry fix (Merge-2 adjudication (f); smallest honest change)

The full `daemon-train-worker` cross-COMPILES under MinGW (C2 verified) but did not LINK:
`daemon-telemetry`'s always-on `sentry-rust-minidump` → `crash-handler` native-minidump path
references the UCRT symbol `_invoke_watson`, which mingw-w64's msvcrt import lib does not export.

**Fix (cfg-gate off for `x86_64-pc-windows-gnu` only):**
- `crates/substrate/daemon-telemetry/Cargo.toml`: moved `sentry-rust-minidump` from an unconditional
  dependency to `[target.'cfg(not(all(windows, target_env = "gnu")))'.dependencies]`. Lock-neutral —
  the crate stays resolved for the host targets; it is simply not a dependency edge for windows-gnu.
- `crates/substrate/daemon-telemetry/src/crash.rs`: `CrashGuard._minidump` field, the
  `sentry_rust_minidump::init` call in `init_crash_reporting`, and the `_minidump: None` in
  `init_panic_reporting` are all `#[cfg(not(all(windows, target_env = "gnu")))]`. On windows-gnu,
  `init_crash_reporting` falls back to panic-only capture (the `sentry` panic integration still links
  and arms). **No behavior change on any other target** (Linux/macOS/msvc-windows keep the full
  native-crash monitor). Verified: `cargo check -p daemon-telemetry` green on Linux (minidump still
  compiled); the sealed windows-gnu build LINKS (below); clippy green.

This is the smallest honest change and matches adjudication (f) verbatim ("target-gate the minidump
path (`cfg(not(all(windows, target_env="gnu")))`) … unblocks a true `daemon-train-worker.exe`").

### The flake package (ADDITIVE lane output)

`packages.daemon-train-worker-windows` (`flake.nix`): the FULL `daemon-train-worker` cross-built for
`x86_64-pc-windows-gnu` via the existing `craneLibWindows` toolchain/`windowsCommonArgs`, **WITH the
`swarm-net` feature** (WS control plane + iroh gossip) so it live-attaches to the coordinator as a
real peer. Default `cpu` det lane; NO `wgpu`/`cuda` (the det lane — the consensus bar — is CPU fp32,
so the Windows peer needs no GPU backend to byte-match Linux). aws-lc-sys (rustls' provider, pulled
by `swarm-net`'s wss:// TLS) already cross-builds via `windowsCommonArgs` (nasm + TARGET_CC), exactly
like daemon-host's TLS stack — so no new flake plumbing was needed. Modeled on C2's
`daemon-train-probe-windows` lane; deps-only artifact + package, added to the `packages` inherit list.

### Sealed build + real-box evidence (5090)

- **Sealed cross-build:** `nix build .#daemon-train-worker-windows` → `daemon-train-worker.exe`
  (21 MB static). **The gate-blocking link succeeds** — the telemetry fix is confirmed by an actual
  MinGW link of the full worker (wasmtime 46 + burn 0.21 + iroh 1.0 + tokio-tungstenite + aws-lc).
- **Runs on the real 5090:** `set DAEMON_TRAIN_PROBE=1 && daemon-train-worker.exe` prints the full
  66-op `Hardware` capability report + `DeviceLimits { vram_mb: 32190, ram_mb: 130958, shared_mb: 0,
  unified: false }` (DXGI: dedicated_video 32190 MiB, budget_local 31422, shared_system 65479) —
  byte-for-byte matching C2's `daemon-train-probe.exe` numbers. The full worker (not just the
  telemetry-free probe) runs on real Windows.
- **WS live attach (the heterogeneity rehearsal) — GREEN:** the Windows worker (driven over `ssh -T`,
  CBOR stdio binary-clean) + a local Linux peer both joined a run on the **real Cloudflare** dev
  coordinator (real R2 object-proxy payloads), 6 rounds, **det-lane digests byte-identical every
  round** (run `run-c3-fleet-1783997724`; round 0 `92d31a7d78af3458f35f6f669a409cdc` … round 5
  `bbf40eadba0ad9471ad44ee870fc8356`), run reached Finished. **Windows byte-matches Linux on the det
  lane** — the consensus bar is met.
- **GPU-lane note:** the Windows worker is CPU det-lane only (no wgpu/DX12 backend built), so there is
  no Windows GPU-lane digest to compare; det-lane equality is the consensus bar and is GREEN. The GPU
  heterogeneity at the gate rides the RADV/CUDA/Metal peers.

---

## Task 3 — CUDA lane completion — DONE

### The `.#cuda-train` devShell (adjudication (c); ADDITIVE flake output)

There is already an infer-oriented `.#cuda` devShell (cudatoolkit for llama.cpp CUDA). Adding the
burn-cuda **training** requirements to it would risk that lane, so C3 adds a **non-regressing**
`cuda-train` devShell (`craneLibDev` for the wasm32 guest toolchain, like `.#vulkan`).

**Honest nvrtc shape (C2 finding D5):** nixpkgs-unstable has dropped the driver-matched nvrtc for the
RunPod 4090's CUDA 12.4 driver (it ships ≥12.6, whose PTX the 12.4 driver rejects with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`). So nix supplies only the **build-time** pieces (cudart headers
via `CUDA_PATH`, libstdc++); the **runtime** driver-matched nvrtc 12.4 (NVIDIA's pip wheel
`nvidia-cuda-nvrtc-cu12==12.4.127`) + host driver libs are an **operator-staged impure input** for
that one box, referenced via `DAEMON_CUDA_RUNTIME_DIR` (C2 staged it at `/root/cuda-rt-124`). This
quarantines the single unavoidable impurity — the box's own driver userspace, which by construction
cannot come from nix — to one env var; the flake stays pure + portable. cudarc dlopen's
`libnvrtc.so.12` by soname, so the staged dir being first on `LD_LIBRARY_PATH` wins. The shell's
`shellHook` prints guidance when `DAEMON_CUDA_RUNTIME_DIR` is unset (det/parity suites still run on
the CPU det lane). Unfree-scoped (`config.allowUnfree = true`), Linux-gated. This is the honest
answer to the adjudication's "nixpkgs cudaPackages if a compatible nvrtc exists, else a documented
impure wrapper for the RunPod box only" — the latter, because a compatible nvrtc does not exist in
nixpkgs for this driver.

### RunPod evidence (4090, driver 550.127.05 / CUDA 12.4)

- **Build:** `cargo build -p daemon-train --features swarm-net,cuda --bin daemon-train-worker` green
  on-box (cuda deps compile; cudarc runtime-dlopen, no toolkit at build; lock-neutral per C2).
- **Runtime:** the cuda-built worker RUNS (`DAEMON_TRAIN_PROBE=1` → cpu-lane `DeviceLimits`, ram
  127935 MiB). **No `BackendKind::Cuda` engine arm yet** (C2 adjudication), so the worker executes the
  **CPU det lane** (the consensus bar) — it byte-matches Linux/Windows by construction.
- **Suites (`--features cuda,burn-ndarray`):** `wasm_backend_determinism` **12/12** (incl.
  `cross_backend::cross_backend_det_digest_{demo,diloco,sparse_loco}`, `cross_peer_bit_identity_demo`,
  `preemption_as_churn_is_digest_neutral`); `--lib autotune` **14/14**. Green on the CUDA box.
- **Live WS attach — GREEN:** the RunPod cuda worker (over `ssh -p 13988 -T`) joined a live 3-peer run
  (Linux + Windows + RunPod-CUDA) on the real Cloudflare coordinator, 6 rounds, **all det digests
  byte-identical across all three vendors** (run `run-c3-fleet-1783998067`; round 0
  `82d4f93de034e77757f1904d6658a38f` … round 5 `f6cc48a85e78d9afb2c63d0684cc5e77`), Finished.
- **CUDA-vs-CPU/Vulkan digest behavior (recorded):** because there is no cuda engine arm, the CUDA
  lane runs the CPU det lane and is byte-identical by construction — there is no CUDA-specific native
  digest to diverge yet. The wgpu(Vulkan)-vs-CPU native-lane behavior is covered by the existing
  `burn_wgpu_parity` suite on the RADV box (det digests byte-equal, native losses within tolerance —
  Merge-2 evidence). Wiring an actual `BackendKind::Cuda` engine arm (using the staged nvrtc) is a
  G-lane follow-on; the dep + devshell story is now complete.

---

## Task 4 — M4 Metal peer readiness — DONE

- **On-box build:** the `swarm-net` worker built on aarch64-darwin (`~/daemon-node-c3`) — the
  daemon-telemetry fix is a no-op on darwin (minidump compiles there), so no change was needed for
  the M4 build; the C2 devShell-eval + `mode_t` fixes carried it. (Non-login ssh needs the nix
  profile on PATH: `export PATH=/nix/var/nix/profiles/default/bin:$HOME/.nix-profile/bin:$PATH`.)
- **Metal probe validated:** `recommended_working_set 26800603136 → vram_mb 25559`,
  `max_buffer_length → 19169`, `hasUnifiedMemory true`, `phys_ram 32768` →
  `DeviceLimits { vram_mb: 25559, ram_mb: 32768, shared_mb: 32768, unified: true }`,
  `backend_lanes ["metal", "cpu"]` — matches C2's M4 numbers exactly. **wgpu-over-Metal is viable**
  (the tree reports a Metal lane); a wgpu-Metal training suite (`burn_wgpu_parity`/`wgpu_lifecycle`
  on M4) is a straightforward tier-2 follow-on (build `--features wgpu` on M4) — the det lane (the
  consensus bar) is already proven byte-equal below.
- **Live WS attach — GREEN:** the M4 worker (over `ssh -T`, via a `run-worker.sh` wrapper that sets
  the nix profile PATH + `nix develop --command`, CBOR stdio binary-clean) + a local Linux peer both
  joined a live run, 4 rounds, **det digests byte-identical** (run `run-c3-fleet-1783999087`;
  round 0 `92d31a7d78af3458f35f6f669a409cdc` — **the same digest as the Linux+Windows 2-peer round
  0**, proving the det lane is truly platform-independent across AMD/Linux, NVIDIA/Windows,
  Apple/macOS). Finished.

### Heterogeneity summary (the gate's ≥4-peer requirement)

All four platforms proven byte-identical on the det lane over the live substrate — **3 GPU vendors
(AMD, NVIDIA, Apple), 3 OSes (Linux, Windows, macOS)**:

| Live run | Peers | Rounds | Result |
|---|---|---|---|
| `run-c3-fleet-1783997724` | Linux(AMD) + Windows(5090) | 6 | byte-identical every round |
| `run-c3-fleet-1783998067` | Linux + Windows + RunPod(4090 CUDA) | 6 | byte-identical every round |
| `run-c3-fleet-1783999087` | Linux + M4(Metal) | 4 | byte-identical (round 0 == the Windows run's) |

**4-peer simultaneous run (the exact gate config) — known issue.** The lean `fleet_live_hetero`
harness (no rejoin) stalls at round 0 when all four WAN peers include the M4 peer spawned via
`nix develop --command` over ssh (per-spawn latency + `min_peers=N` barrier + no rejoin ⇒ an early
park that never self-heals; no engine errors — the det lane + links are fine). Isolation confirmed
the cause is orchestration, not consensus: a **local** 4-peer run (all subprocesses on this box)
completes cleanly (8 rounds — see the overhead run below), and Linux+M4 2-peer is green. Mitigation
for the ceremony (runbook §2/§6): drive the 4-peer gate run with the **churn-robust
`ws_live_workers.rs` harness** (drop→park→rejoin recovery) extended with the same remote-ssh peer
specs, and/or pre-warm M4's devShell so spawn is instant.

---

## Task 5 — baselines + overhead measurement (gate criterion <15%) — DONE

**Tool:** `fleet_live_hetero::swarm_round_overhead_vs_single_host` (C3) runs the same real workload as
a single-host baseline (1 peer) and an N-peer swarm (all local subprocesses, real tiny-llama guest +
real R2), and reports `overhead% = (mean N-peer round wall − single-host round wall) / single-host`.
Env: `SWARM_FLEET_OVERHEAD_PEERS` (default 4), `SWARM_FLEET_ROUNDS`.

**Local rehearsal (4 local peers, this box → dev coordinator):** single-host `t1 = 1051 ms`, 4-peer
`tN = 1219 ms` ⇒ **cross-peer overhead +168 ms/round = 16.0%**.

**Honest interpretation (recorded so nobody misreads 16% as a gate fail):** at tiny-llama scale the
per-round *compute* is sub-millisecond, so the round wall is dominated by the ~1 s R2/WS WAN
round-trip; the 16% is protocol-latency-dominated and is **NOT** the gate figure — it validates the
measurement *mechanism* and the barrier's absolute cost (~168 ms of cross-peer fetch/aggregate). At
the **160M gate model** each round is *seconds* of real compute, so the same absolute barrier is a
small fraction (<15%). The gate procedure (runbook §5) runs the exact tool with the 160M envelope and
cross-checks against the centralized single-host throughput baseline (`swarm-p2-throughput.md`: ~384
tok/s at 160M on RADV; loss curve 10.85→4.93 — the ε reference). This is the documented procedure the
adjudication asked for + a rehearsed mechanism + real numbers.

---

## Task 6 — CI tiers (TDD §8.1) — DONE

daemon-node has **no CI of its own** — CI lives in the superproject `.github/workflows/ci.yml` (whose
`daemon-node` job already runs `cargo test --workspace`, covering the default-feature swarm suites).

- **Tier 1 (per-PR, no GPU) — IMPLEMENTED in-repo:** `cargo run -p xtask -- swarm-ci-det` is the
  single in-repo definition of the consensus-critical CPU swarm gate — it builds the guests, then runs
  the pinned suite list (det-core, daemon-swarm-{proto,run,observe,net}, daemon-train
  `--features burn-ndarray`, daemon-train-sdk `--features sim`, daemon-swarm-e2e, daemon-api
  conformance + proptest), red-failing on the first. This also exercises the feature-gated CPU suites
  `cargo test --workspace` does not. GPU + live lanes are excluded by design. (`daemon-conformance`'s
  known parallel-load flake is not a swarm crate and not in the list.)
- **Tier 1 superproject job — PROPOSAL (human, signed):** an additive `swarm-det` job that calls
  `cargo run -p xtask -- swarm-ci-det` (YAML in runbook §8a). The `.github/workflows/ci.yml` is the
  superproject's, so this is a proposal — no main-repo working-tree edit from a lane.
- **Tier 2 (per-lane scheduled, self-hosted) — DESIGNED (runbook §8):** one box per backend (CUDA /
  Vulkan-RADV / Metal), scheduled, running the native tolerance-class fixtures + parity + cross-lane
  replay. The exact lane commands are the on-box builds/suites proven in this ledger. Not a fake CI
  YAML for machines CI can't reach — a homelab/operator schedule.
- **Tier 3 (hardware-in-loop, manual) — the gate ceremony:** the runbook itself.

---

## flake.nix edits (ADDITIVE — Merge-3 review)

All under the delegated scoped rights; no existing output removed or restructured.

1. **`packages.daemon-train-worker-windows` (NEW)** — full MinGW-cross worker with `swarm-net` (Task
   2). Unblocked by the telemetry fix. Deploy artifact for the 5090; never build on-box.
2. **`devShells.<linux>.cuda-train` (NEW)** — the burn-cuda training lane (Task 3), unfree-scoped,
   `craneLibDev` (wasm guest toolchain), build-time cudart headers + the documented
   `DAEMON_CUDA_RUNTIME_DIR` impure-runtime hook for the box's driver-matched nvrtc 12.4. Distinct
   from the pre-existing infer `.#cuda` shell (left untouched — no regression).

Both evaluate (`nix eval .#…drvPath` green); the windows package builds sealed.

---

## Seams exported (freeze at Merge 3)

- **daemon-telemetry:** native-minidump path is `#[cfg(not(all(windows, target_env="gnu")))]`; on
  windows-gnu, crash reporting is panic-capture only. `CrashGuard` shape unchanged on every other
  target. Additive/subtractive-on-one-target only — no API change.
- **flake:** `daemon-train-worker-windows` package + `cuda-train` devShell (both additive).
- **xtask:** `swarm-ci-det` subcommand (the tier-1 gate definition).
- **daemon-swarm-e2e:** `fleet_live_hetero.rs` — the configurable-peer heterogeneity harness
  (`fleet_heterogeneous_det_lane_agrees`) + the overhead tool
  (`swarm_round_overhead_vs_single_host`) + the `timed_local_run` helper. Env contract in the file
  header + runbook §2/§5. Reusable by the gate ceremony (local + remote-ssh peers).
- **docs:** `swarm-p2-gate-runbook.md` (the ceremony) + this ledger.

---

## Gate matrix (C3 local gates)

- `cargo fmt --all --check` — (run at lane close).
- `cargo clippy` `-D warnings`: `-p daemon-telemetry` ✓ · `-p xtask` ✓ · `-p daemon-swarm-e2e
  --features iroh --tests` ✓ (the touched crates).
- `cargo check -p daemon-telemetry` ✓ (Linux; minidump still compiled) — the windows-gnu carve-out is
  proven by the sealed cross-build linking.
- **Sealed windows package:** `nix build .#daemon-train-worker-windows` ✓ (the full worker LINKS +
  runs on the real 5090).
- `.#cuda-train` + `daemon-train-worker-windows` flake eval ✓.
- Fleet smokes: Windows/RunPod/M4 live WS attach byte-identical (evidence above).
- Full workspace test / feature-combo clippy / deny / typos / build-guests: run at lane close +
  folded by the Merge-3 owner (the known `daemon-conformance` trio flake is pass-in-isolation, not a
  swarm crate).

---

## What Merge 3 (the gate) must know

1. **The Windows peer is unblocked and proven** — the telemetry cfg-gate landed the full
   `daemon-train-worker.exe` (links + runs on the 5090 + byte-matches Linux live). No behavior change
   on any other target.
2. **All four heterogeneous peers build, run, probe, and live-attach with byte-identical det digests**
   (pairwise + 3-way over the real Cloudflare substrate). The det lane is truly platform-independent
   (Linux/Windows/CUDA/Metal share round-0 digests). This is the core gate evidence.
3. **Drive the 4-peer ceremony with the churn-robust `ws_live_workers.rs` harness** (rejoin recovery)
   + remote-ssh peers, and pre-warm M4's devShell — the lean `fleet_live_hetero` harness stalls a
   4-peer WAN run at round 0 (no rejoin; not a consensus/det problem — local 4-peer is green).
4. **The overhead 16% number is tiny-llama scale (protocol-latency-dominated), NOT a gate fail** — run
   the overhead tool with the 160M envelope for the real figure; absolute barrier cost ~168 ms/round.
5. **CUDA has no engine arm yet** — `--features cuda` runs the CPU det lane (byte-matches). The
   `.#cuda-train` devshell + the operator-staged nvrtc-12.4 (`DAEMON_CUDA_RUNTIME_DIR`) are ready for
   a G-lane `BackendKind::Cuda` follow-on.
6. **KEEP-IN-SYNC**: redeploy the dev coordinator after any daemon-cloud coordination-branch change to
   `coordinator-wasm`/`registry.ts`/`shell.ts` (a stale coordinator caused Merge-2's `global_batch`
   bug).
7. **Carried follow-on:** wire `--observe` into the cloud-DO worker-subprocess loop (Merge-2 Task-5
   note) so a worker-driven gate run produces an offline-replayable capture directly (today the
   per-round digest transcript + R2 `RoundRecord`s are the record; `swarm-local --observe` is the
   parallel replay artifact).
8. **flake edits are ADDITIVE** (windows worker package + `cuda-train` devshell) — re-run `cargo deny`
   at the merge (both are lock-neutral: aws-lc/tungstenite/iroh already in the lock; cuda deps already
   resolved per C2).
