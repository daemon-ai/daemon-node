# Swarm P2 WAN — program ledger

Wave-0 scaffold coordination record for the **Swarm P2 WAN Program** — taking the landed P1
protocol stack from in-process shells to a real WAN swarm and passing the spec's P2 research gate
([swarm-training-spec.md §17](swarm-training-spec.md)): 160M–500M training across ≥4 heterogeneous
consumer GPUs (incl. one ROCm/Vulkan peer) with forced churn, ε-convergence vs a centralized
baseline, and <15% round overhead. This is the single source of truth for the trunk, lane
file-ownership, the frozen-file rule, the inherited conventions, and the fleet inventory. Lane agents
working in a P2 worktree: **read this before you touch anything** — it carries everything you need
without reaching into `~/.cursor`.

This ledger governs P2 on top of the completed P1 program. The P1 program record
(`swarm-p1-ledger.md`) — its Merge-1/2/3 **frozen-interfaces** sections and the MVP-era
`swarm-mvp-ledger.md` — remain authoritative for every seam frozen at or before P1. **P2 inherits
P1's frozen seams: extend them additively only.** In particular `tabi@1` (the 66-op tensor ABI) is
**FROZEN FOREVER** at the P1 exit gate (spec §16) — additive `op@version` growth only; a breaking
change is `tabi@2`.

## Program goal — the P2 WAN gate (program exit)

1. **Centralized baseline:** a 160M (and 500M if fleet VRAM allows) single-host reference run — reuse
   the P1 M2 parity harness for the loss curve.
2. **WAN run:** ≥4 heterogeneous peers (fleet incl. Strix Halo/Vulkan, M1/Metal, CUDA + ROCm boxes),
   cloud DO coordinator, real R2, self-hosted relay; forced churn drills mid-run.
3. **Pass criteria:** ε-convergence vs baseline, zero det-digest mismatches, round overhead <15%
   (barrier ingest gap), replay oracle green over the full run log.

The user confirmed the **full** gate (hardware available); the app UI surface (WIRE-4) is deferred to
its own **P3** program.

## Base + trunk

- **Repo:** `daemon-node` (Rust backend submodule; standalone checkout).
- **Base commit:** `1869532` (`test(store,telemetry): stamp migration ladder at 17 + crash-reporting
  smoke example`) — master **moved past the P1 landing**: it now carries the crash-reporting work,
  including a **WireVersion bump to v41** at `1eaf6c9` (`feat(api): crash-reporting consent wire API
  (wire v41)`) and a devShell change at `76f9a7a` (`fix(devshell): pin CARGO_TARGET_DIR to the
  checkout`). P1 landed on `integrations/swarm-p1` (kept as the historical record; final content HEAD
  `1c2c43a`). daemon-cloud `daemon-api` master = `0482a68` (the P1 `swarm/bc` landing, unmoved).
- **Trunk:** `integrations/swarm-p2` (one shared trunk, forked from `1869532`), worktree at
  `/home/j/experiments/daemon-worktree/swarm-p2-integration`. A/B/C lanes interleave wave-wise and
  merge back here; the integration owner owns the frozen files, the `WireVersion` bump, seam swaps,
  and this ledger.
- **Worktrees:** each lane subagent works EXCLUSIVELY in its assigned worktree under
  `/home/j/experiments/daemon-worktree/` (daemon-node) or a branch checkout of
  `/home/j/experiments/daemon-cloud/daemon-api` (the C1 cloud lane). Never modify the main checkouts,
  never `git push`, never `--no-verify`.

### Wire version note (READ THIS)

`WireVersion::CURRENT` is **v41** on the moved master (the additive crash-consent surface;
`contract_wire_version_is_v41`). The P1 `SwarmApi` entries (v40) are **historical**. **Any P2 wire
change is additive and targets v42** — bump `WireVersion` 41→42 **once**, in a single coordinated
integration-owner commit at the merge that introduces it (mirroring the P1 39→40 discipline), with
`just update-codec` + `codec-drift` green (the superproject half is a human, signed step). Candidate
P2 wire additions: `SwarmHardwareReport.shared_mb` (GUI mirror of the worker `Hardware.shared_mb`),
`Metric`/`Warning` telemetry events, and any C2 `Hardware` platform fields. Eligibility stays
node-computed; telemetry stays fixed-point integers on the wire (no floats).

## Waves & lanes (from the program plan)

Three lanes max per wave, as in P1. Merge owners freeze the named seams at each merge.

| Wave | Lane | Scope (disjoint by construction) | Merge freeze |
|---|---|---|---|
| **1** | **A1** node WS coordinator client | new module in `daemon-swarm-net`: consume `GET {base}/runs/:id/ws` on the RunCoordinatorDO, canonical-CBOR `SignedMessage` frames, dedupe WS+gossip via `Deduper`; discovery via registry `GET /runs`; envelope fetch + worker `AssessRun` on join | WS-client seam (Merge 1) |
| **1** | **A4 + A2 (cloud half)** | wasm32 `tick` inside the RunCoordinatorDO (COORD-3 build green) → `dual_shell_parity` vs `LocalCoordinator`; `HttpPresignClient` at `apps/swarm` (wrangler-dev → real bucket, SigV4); `swarm-local --store r2` | presign-live seam (Merge 1) |
| **1** | **B1** sparse_loco flagship | SDK-1..5 goldens; HOST-1/2 (DCT, topk_chunk), HOST-5/6/7 det-lane suites; PROTO-8/9/10, PROTO-7 unit IDs, RUN-6/7/10, CLI-2..4 | — |
| **2** | **A3** worker live attach | add `daemon-swarm-net` (iroh + WS) to the worker behind its own feature; replace self-driven `JoinRun` round with `RoundEngine` over `IrohGossip`+`R2Store`; coordinator frames from `JoinRun.coordinator`; continuous `TrainSupervisor` event pump → `SwarmService::handle_worker_event`; additive `Metric`/`Warning` telemetry | worker attach seam (Merge 2) |
| **2** | **B2 + B3** | wire `daemon-swarm-observe` (MessageLog, replay oracle, desync tally, run health) into the live runtime + `swarm-local`; lazy device-resident `OpBackend` results (remove the 2.3–2.6× host-copy tax) | observe + OpBackend (Merge 2) |
| **2** | **C1 + C2** | CUDA + ROCm worker flake lanes beside `.#vulkan` + per-lane tolerance fixtures + cross-lane replay; Windows DXGI/D3D12 probe FFI (`windows = "0.62"`, worker-only), macOS Metal `recommendedMaxWorkingSetSize` probe, `Hardware.shared_mb` additive DTO | GPU lanes + platform probes (Merge 2) |
| **3** | **C3 + gate prep** | per-lane scheduled runners, hardware-in-loop gate runbook; baseline runs, fleet provisioning; remaining B suites | — |
| **Gate** | **Merge 3** | the P2 WAN gate ceremony (≥4 heterogeneous peers, churn, ε-convergence, <15% overhead, replay green) | program exit |

**Cross-lane dependency edges** are wired via `[workspace.dependencies]` path entries: a lane
consuming another lane's crate uses `{ workspace = true }` and does **not** edit that crate. Adding a
**new member crate** is fine (the `crates/*/*` glob picks it up with no root edit). Adding a **new
third-party dependency** (or a feature of a workspace dep that pulls a new crate) is NOT a lane action
— request it from the integration owner (who re-runs `cargo deny check`). Adding a *feature* of an
already-declared workspace dep from your own crate's `Cargo.toml` (e.g. `tokio-tungstenite`'s TLS
feature, or `daemon-swarm-net/iroh`) IS a lane-owned edit.

## Conventions inherited from P1 (carry over verbatim unless noted)

- **Worktree ownership** — one lane, one worktree; disjoint file ownership is the merge guarantee. The
  integration owner owns the frozen files, the WireVersion bump, seam swaps, and this ledger.
- **FROZEN files (single-writer, integration-owner only):** root **`Cargo.toml`** (workspace members
  glob, `exclude = ["guests"]`, `[workspace.dependencies]`, `[workspace.lints]`, profiles),
  **`deny.toml`** (advisory/license/ban/source policy), **`flake.nix`** (devShell toolchain + targets
  + package/devShell lanes). Repo-root `.gitleaks.toml` / `typos.toml` are NOT lane-frozen.
- **FROZEN interfaces (extend additively only):** everything frozen through P1 Merge-3 — `tabi@1`
  (66 ops, **frozen forever**), the reference-parity harness API, `LiveSwarmConfig`/`run_live_swarm`,
  the `set_swarm`/`emit_node_event` boot shape + `[swarm]` config, worker lifecycle glue, the
  three-platform `DeviceLimits` probe design, the node↔cloud presign/WS HTTP contract
  (`tests/fixtures/presign-*.json`, byte-frozen). See `swarm-p1-ledger.md` Merge-1/2/3
  "Frozen interfaces".
- **Commit styles:** `feat(...)`/`fix(...)`/`build(workspace|deps|nix)/(...)` per change; **lane
  ledgers** land as `mirror(<lane>): ...`; **integration/merge** records + this program ledger land as
  `mirror(wave-N|merge-N): ...`. Merges are `--no-ff` (ort); the disjoint ownership keeps `Cargo.lock`
  the only co-touched file (git auto-merges the additive regions).
- **Mirror commits:** each lane's `docs/specs/swarm-ledger-<lane>.md` is its own file (no collisions);
  the integration owner folds seam records into this ledger at each merge.
- **daemon-node does NOT sign** (`commit.gpgsign=false` by submodule convention). The **superproject**
  requires GPG-signed commits with explicit human approval — every superproject change here is a
  *proposal for the human*, never committed by an agent.
- **Billing-stall note (operational, carried from P1):** long nix builds and multi-minute
  `cargo test`/`nix build` runs produce no output for stretches (the P1 workspace test is ~6 min; the
  release wgpu 160M parity is ~4–6 min cold). A silent long build is **expected, not hung** — background
  long operations with a completion sentinel and monitor them; do not treat a quiet multi-minute
  builder as a stall, and do not stack a second build against it (resource discipline: one build at a
  time, jobs capped at ≤ nproc/2).
- **Disk hygiene:** on lane completion, reclaim the finished lane's build artifacts with
  `/home/j/experiments/daemon-worktree/clean-lane-target.sh <worktree-dir>` (deletes ONLY `target/`
  and `guests/target/`; refuses on uncommitted tracked changes or an active cargo/rustc build). Prune
  lane targets between waves to keep `$HOME` off the swap cliff.
- **`build-guests` after every checkout (P1's hardest-won lesson):** the wasm guests live under the
  gitignored `guests/target/**` and do NOT travel with a branch. **ALWAYS run
  `cargo run -p xtask -- build-guests` in a fresh worktree before any wasm-backed test** — a
  stale/missing guest used to surface as a silent NaN. P2 hardens this into a loud guard (see
  "Carried items → stale-guest guard" below).
- **Known flake — never modify:** the `daemon-conformance` detached-delegation/operator-steer trio is
  nondeterministic under full parallel load; **pass-in-isolation = green**
  (`cargo test -p daemon-conformance --lib node::detached_delegation`). No swarm lane touches
  `daemon-conformance`. (This wave-0 run happened to be fully green.)

## Fleet inventory (procured, pinned 2026-07-13)

A sibling agent is provisioning these machines and will produce
`/home/j/experiments/daemon-worktree/fleet-report-p2.md` (iroh-relay RTTs, per-box Nix status, CUDA
smoke). **Reference that report — do not wait for it.** Totals: **5 peers, 3 GPU vendors (AMD, Apple,
NVIDIA), 3 OSes (Linux, macOS, Windows)** — exceeds the gate's ≥4 heterogeneous requirement incl. the
ROCm/Vulkan peer.

| Peer | Access | GPU / backend | Memory | Role | Status / Wave-0 setup |
|---|---|---|---|---|---|
| **Strix Halo** (this box) | local | AMD Ryzen AI Max+ 395, RADV/Vulkan, UMA | 128 GB UMA | the required **ROCm/Vulkan** peer; the RADV GPU test lane (`.#vulkan`) | Ready |
| **M1 Mac mini** | `m1@51.159.120.241` | Apple Metal | — | macOS/Metal peer | Ready (probe-validated in P1); Nix installed |
| **M4 Mac** | `m1@62.210.193.129` | Apple Metal, macOS 26.3.2 | 32 GB | macOS/Metal peer | **Needs Nix** (Determinate installer; sudo TBC) → clone + verify `nix develop` on aarch64-darwin |
| **RunPod RTX 4090** | `ssh.runpod.io` proxy (needs direct TCP) | NVIDIA CUDA (Ada) | 61 GB RAM, `/workspace` 500 GB | Linux **CUDA** peer | **User action: enable direct TCP SSH** (proxy needs interactive PTY, blocks agent exec); single-user Nix under `/workspace` (ephemeral container, no systemd); CUDA smoke (`nvidia-smi`, wgpu adapter probe) |
| **Windows Server 2022 + RTX 5090** | `ssh usergpu356@37.230.134.194` | NVIDIA CUDA + Vulkan 1.4.341 (5090, 32 GB) | — | Windows CUDA/Vulkan peer **and** the real-Windows DXGI/UMA probe validation box | Driver 610.74 OK; **deploy MinGW cross-built worker binaries, never build on-box**; run DXGI probe FFI + manual Task-Manager cross-check. NB: stale phantom PnP entries for a 4090/2×3090 from a prior tenant — only the 5090 is attached |

All peers: iroh-relay reachability check against the self-hosted relay; record RTTs in the fleet
report.

---

## Wave-0 scaffold record

Landed on `integrations/swarm-p2` (base `1869532`). Commit list (oldest → newest; this
`mirror(wave-0)` ledger commit sits on top):

| Commit | Subject |
|---|---|
| `f86c29a` | `fix(guests): build guest wasm into guests/target under devShell CARGO_TARGET_DIR` |
| `40eef23` | `build(workspace): reserve windows 0.62 workspace dep for the C2 Windows VRAM probe` |
| `d2d936c` | `feat(guests): committed blake3 manifest + stale-guest harness guard` |
| `mirror(wave-0): program ledger` | this ledger |

### The "master moved" integration check — one real break found + fixed (`f86c29a`)

The crash-reporting master's `76f9a7a` (`fix(devshell): pin CARGO_TARGET_DIR to the checkout`) exports
`CARGO_TARGET_DIR=<checkout>/target` in the devShell. That env var **leaked into the `guests/`
mini-workspace build** (a *separate* cargo workspace), redirecting the guest `.wasm` to
`<checkout>/target/wasm32-.../release` while every test harness reads `guests/target/wasm32-.../release`
— so all wasm-backed suites failed with `No such file` (a *harder* failure than the P1 NaN, but the
same class of stale/missing-guest bug). **Fix:** clear `CARGO_TARGET_DIR` when shelling the guests
build so cargo defaults to `guests/target` — in `xtask build-guests` and all nine test-harness
`ensure_built()` copies. After the fix the full swarm stack passes on the moved master (see gate
results). **No other swarm suite was broken by the crash-reporting master.**

Verification on the moved master (all green): `build-guests`; `cargo test -p daemon-swarm-e2e`
(swarm_e2e 2, wasm_profiles 3, drills 5); `cargo test -p daemon-train --features burn-ndarray`
(guest_lifecycle 9, wasm_backend_determinism 12, worker_protocol 4, burn_backend_parity, abi_surface,
…); `cargo test -p daemon-swarm-net` (67 + conformance 2) and `--features iroh` (conformance 4,
iroh_gossip 7); `--features iroh --test live_transport` 6/6 (incl. tiny-llama-over-iroh + self-hosted
relay); fmt + `cargo clippy --workspace --all-targets -D warnings`.

### Frozen-file pass (integration-owner scope)

- **Root `Cargo.toml` (`40eef23`) — ONE edit:** added `windows = "0.62"` to `[workspace.dependencies]`
  for the **C2** Windows VRAM/UMA probe (`swarm-windows-vram-design.md §4`; worker bin only). The
  entry is **inert until the C2 lane references it** — `Cargo.lock` is unchanged and `cargo deny` stays
  green (the `windows`/`windows-sys` crate family is already resolved in-tree, MIT OR Apache-2.0,
  already allow-listed). The C2 lane wires it under `[target.'cfg(windows)'.dependencies]` in the
  worker crate and selects the API-module features there (design §4: `Win32_Foundation`,
  `Win32_Graphics_{Dxgi,Dxgi_Common,Direct3D,Direct3D12}`, `Win32_System_SystemInformation`, +
  `Win32_System_Threading` only if the budget-notification event is wired).
- **WS-stack decision for A1 (NO root edit needed):** the tree already carries **`tokio-tungstenite
  0.29`** as a `[workspace.dependencies]` entry, already used by `daemon-host` (`ws.rs`, the
  browser/WASM CBOR-mux WS carrier). **A1 reuses that stack — do NOT add a second WS library.** The
  root entry has no TLS features on purpose (the node's own WS server terminates `wss://` at a reverse
  proxy). A1's client connects *out* to the cloud RunCoordinatorDO over `wss://`, so it enables the
  rustls TLS feature **in `daemon-swarm-net`'s own `Cargo.toml`** —
  `tokio-tungstenite = { workspace = true, features = ["rustls-tls-webpki-roots"] }` (a lane-owned
  feature edit) — matching the tree's rustls/aws-lc posture (no native-tls anywhere; `tokio-rustls`
  0.26 already in-tree). **Gate it behind a `daemon-swarm-net` cargo feature** (like `iroh`) so the
  default gate stays TLS/WS-client-free; the integration owner re-runs `cargo deny` when A1 lands
  (the feature may pull `webpki-roots`). Discovery (`GET /runs`) + envelope fetch ride
  `daemon_egress::EgressClient` (raw `reqwest` is clippy-banned).
- **Other Wave-1/2 workspace deps — none missing.** All swarm path crates are already
  `[workspace.dependencies]` entries, so **A3** adds `daemon-swarm-net` to the worker (`daemon-train`)
  as `{ workspace = true, optional = true }` behind its own feature (no root edit), and **B2** consumes
  the existing `daemon-swarm-observe` entry.
- **`deny.toml` — no change (recorded verdict).** `cargo deny check` is fully green as-is
  (advisories/bans/licenses/sources ok). The `windows` entry adds nothing to the lock yet; when C2
  wires it, the `windows` family is already permissive + in-tree. Re-run `deny` at the C2/A1 merges
  (A1's WS TLS feature, C1's CUDA/ROCm trees).
- **`flake.nix` — no change this wave; C1 requirements recorded (not a cheap stanza).** There is **no
  daemon-infer CUDA precedent** — the engine flake outputs are Vulkan-via-llama.cpp (`.#daemon-infer-vulkan`,
  `flake.nix:418`), Metal (`.#daemon-infer-metal`), and MinGW Windows cross (`daemon-infer-*-windows`,
  `flake.nix:749`); no `cudaPackages`/`config.cudaSupport` anywhere. A real CUDA lane is unfree + heavy
  (nvcc/libcudart/libcublas at build+runtime), so per the "don't build heavy CUDA toolchains" guidance
  it is **NOT** added at Wave 0. **What C1 will need (Wave 2):**
  1. `nixpkgs.config.allowUnfree = true` (scoped) + `cudaPackages` (nvcc, cuda_cudart, libcublas) — an
     unfree-gated import, like the `.#review` shell isolates its unfree `codeql`.
  2. burn's `cuda` cargo feature on `daemon-train` (→ cubecl-cuda), a `.#cuda` devShell/package lane
     mirroring the `.#vulkan` pattern, with `CUDA_ROOT`/`LD_LIBRARY_PATH` wired for `libcudart`.
  3. **ROCm:** the gate's "ROCm/Vulkan peer" is already satisfied by **Vulkan on RADV** (Strix Halo,
     `.#vulkan`). A native ROCm/HIP lane is optional; if pursued, a `.#rocm` shell with `rocmPackages`
     — heavier still. Prefer Vulkan/RADV for the gate's heterogeneity requirement.
  4. Build CUDA lanes on the CUDA boxes (RunPod 4090 / Windows 5090), not on this box — one sealed
     `nix build` at most, at the end (resource discipline).

### Carried items from P1

1. **Stale-guest guard — IMPLEMENTED (`d2d936c`).** `xtask build-guests` now hashes the built modules
   and writes the **committed** `guests/guests.blake3`; every wasm-backed harness (`ensure_built`, 9
   copies) asserts the module it loads matches the manifest — on **both** the rebuild and the
   `SWARM_TEST_GUEST_DIR` prebuilt paths — so a stale/mismatched guest fails **loud** with a
   remediation hint instead of a downstream NaN. To make the committed manifest portable, the guest
   build is now **byte-reproducible**: `build-guests` and the harnesses remap the absolute checkout +
   cargo-registry prefixes rustc bakes into panic locations (`--remap-path-prefix …=/daemon-node`,
   `…=/cargo`), so the hashes are identical across checkouts/machines under the pinned toolchain.
   `SWARM_TEST_GUEST_DIR` still short-circuits the build (CI prebuilt) **and now verifies**. Validated
   positive (prebuilt dir → green) and negative (tampered manifest → loud failure). *(`trim-paths` was
   evaluated and rejected — it still requires nightly cargo on this toolchain.)*
   - **Maintainer workflow:** after changing guest source (or a pinned-toolchain bump that shifts the
     bytes), run `cargo run -p xtask -- build-guests` and **commit `guests/guests.blake3`**. Proposed CI
     drift check (superproject): `build-guests` then `git diff --exit-code -- guests/guests.blake3`.

2. **`just swarm-dev` reconciliation — LEDGER-ONLY diff (justfile is superproject-frozen, human
   applies).** There are **two** binaries named `swarm-local`:
   - `bins/swarm-local` — the **envelope runner** (authoring TOML → freeze/verify §6.1 → in-process
     `--backend stub` or supervised `--backend worker`). No transport selection; frozen at P1 Merge 3.
   - `daemon-swarm-run/src/bin/swarm-local.rs` — the **transport harness** (`--transport
     loopback|iroh --store fs --peers N --rounds N --relay <url>`), drives the real
     `daemon-swarm-coordinator` `tick` loop over a selectable transport, prints the agreed per-round
     digest transcript, and **exits non-zero on divergence**. Requires the `iroh` feature.

   **Decision:** the dev loop drives the **transport harness** — it exercises the real coordinator +
   loopback/iroh transport, is fast (deterministic `StubBackend`, no GPU/guest build), and returns a
   pass/fail exit code (good for `just e2e`). Proposed superproject `justfile` recipe (do NOT apply
   here):
   ```just
   # Local swarm dev loop: in-process N-peer round loop over the real coordinator tick + a selectable
   # transport (iroh real mesh by default; loopback for a GPU-less/offline smoke), fs payload store.
   # Prints the agreed per-round digest transcript; exits non-zero on divergence. Deterministic
   # StubBackend (no GPU/guest build). Drives daemon-swarm-run's transport harness, NOT bins/swarm-local.
   swarm-dev transport="iroh" peers="3" rounds="8":
       cd daemon-node && nix develop --command cargo run -p daemon-swarm-run --features iroh \
         --bin swarm-local -- --transport {{transport}} --store fs --peers {{peers}} --rounds {{rounds}}
   ```
   And wire a loopback smoke into `just e2e` (no relay/GPU needed), per the P1 handoff:
   ```just
   # add to the `e2e` recipe body:
   ( cd daemon-node && nix develop --command cargo run -p daemon-swarm-run --features iroh \
       --bin swarm-local -- --transport loopback --store fs --peers 3 --rounds 8 )
   ```
   `--store r2` remains a BC follow-on (needs the live presign endpoint / wrangler-dev); it is
   rejected with a pointer today.

3. **Spec-amendment proposals — LEDGER-ONLY (spec docs are the human's to edit).** Carried from P1,
   still unapplied:
   - **§10.5 unified-governor clamp (from P1 Merge-2).** On unified-memory boxes the effective device
     budget is `vram_mb + 90%·shared_mb` and device + host compete for one DRAM pool, so the policy
     `vram_cap_mb` (`SwarmPolicy`) is the **only** protection for a co-resident inference tenant — it
     must clamp the **combined** effective budget, not just the dedicated-VRAM term. §10.5 today reads
     as a plain VRAM cap; state the combined-budget semantics explicitly. (Load-bearing for C1/C2 +
     the gate on Strix Halo.)
   - **§5.1 fp32 note (from P1 M1/Merge-2).** The §5.1 planning table assumes bf16 weights (160M row:
     0.3 GB); the P1 preset stores **fp32 masters** for det-lane exactness (0.57 GiB, 2×). Annotate the
     table as bf16-specific + add an fp32-storage note (or adopt bf16 storage in a later G-lane).
   - **`tabi@1` freeze annotation (from P1 Merge-3).** `tabi@1` (66 ops) **froze at the P1 exit gate**,
     HEAD `1c2c43a`, 2026-07-13. The ABI spec §9 describes the freeze *process* but carries no concrete
     `FROZEN AT: <commit/date>` marker; add one. The ledger is authoritative meanwhile.

### Lane worktrees / branches (created this wave)

| Lane | Branch | Location | Base |
|---|---|---|---|
| **A1** | `swarm/a1` | `/home/j/experiments/daemon-worktree/p2-a1` | trunk HEAD |
| **B1** | `swarm/b1` | `/home/j/experiments/daemon-worktree/p2-b1` | trunk HEAD |
| **C1 (cloud, A4/A2-cloud)** | `swarm/c1` | `daemon-cloud/daemon-api` (branch only, no worktree) | daemon-api master `0482a68` |

A3/B2/B3/C2/C3 worktrees are created at their wave launch from the then-current trunk HEAD (Wave 2/3),
same convention. The cloud lane works in `/home/j/experiments/daemon-cloud/daemon-api` on `swarm/c1`;
daemon-cloud is **not gitlinked** to this trunk — the only runtime contract is the presign/WS HTTP
surface (`tests/fixtures/presign-*.json`, byte-frozen from P1) + the WS coordinator endpoint the A1
client consumes. Contract source: `daemon-cloud/daemon-api/docs/swarm-bc-ledger.md`.

### Gate results (Wave-0, trunk HEAD `d2d936c` + this ledger)

All green except the documented pre-existing `daemon-conformance` flake (fully green this run):

- `cargo fmt --all --check` ✓ · `cargo clippy --workspace --all-targets -- -D warnings` ✓.
- Feature-combo clippy `-D warnings`: `-p daemon-train --features burn-ndarray` ✓ · `--features wgpu`
  (`.#vulkan`) ✓ · `-p daemon-swarm-net --features iroh` ✓ · `-p daemon-swarm-run --features iroh` ✓ ·
  `-p daemon-swarm-e2e --features iroh` ✓ · `-p daemon-train-sdk --features sim` ✓ ·
  `-p daemon-api --features arbitrary` ✓.
- `cargo deny check` ✓ (advisories/bans/licenses/sources ok).
- `cargo test --workspace` ✓ (0 failures; the `daemon-conformance` detached-delegation trio was green
  this run).
- Swarm suites: `daemon-swarm-net` default + `--features iroh` ✓ · `daemon-train --features
  burn-ndarray` ✓ · `daemon-swarm-e2e` default ✓ + `--features iroh --test live_transport` **6/6** ✓ ·
  `daemon-train-sdk --features sim` ✓.
- `cargo run -p xtask -- build-guests` ✓ (writes `guests/guests.blake3`) · both
  `wasm32-unknown-unknown` builds (`daemon-swarm-{proto,coordinator}`) ✓ · `typos docs/specs` ✓.
- Stale-guest guard ✓ (positive: `SWARM_TEST_GUEST_DIR` prebuilt → green; negative: tampered manifest
  → loud failure).

### Wave-1 lanes — what to know beyond this ledger

- **A1:** reuse `tokio-tungstenite 0.29` (workspace dep; enable `rustls-tls-webpki-roots` in
  `daemon-swarm-net`'s own manifest behind a new cargo feature; do NOT add a second WS lib). Dedupe
  WS+gossip via the existing `Deduper`. Discovery/envelope-fetch over `daemon_egress::EgressClient`.
  Consume the coordinator WS/DO surface from `swarm-bc-ledger.md`; the presign JSON fixtures are the
  byte-frozen contract. Do NOT bump `WireVersion` in-lane (Merge 1 does 41→42 once, if a wire change
  lands).
- **B1:** default (no-iroh) gate; build the det-lane suites + sparse_loco goldens on the frozen P1
  seams. `build-guests` before wasm suites; the stale-guest guard will fail loud on a stale module.
- **A4/A2 (cloud):** work in `daemon-cloud/daemon-api` on `swarm/c1`; implement the presign endpoint to
  the frozen fixtures verbatim; wasm32 `tick` into the RunCoordinatorDO for `dual_shell_parity`.
- **All lanes:** frozen files (root `Cargo.toml`/`deny.toml`/`flake.nix`) are locked — route new
  third-party deps / features-that-pull-new-crates through the integration owner. Run
  `build-guests` after every checkout/rebase.
