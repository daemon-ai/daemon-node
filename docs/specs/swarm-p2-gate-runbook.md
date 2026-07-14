# Swarm P2 WAN-gate ceremony — operator runbook

The exact, archaeology-free procedure to execute **Merge 3 = the P2 WAN research gate**
([swarm-training-spec.md §17](swarm-training-spec.md); the ledger's older numbering calls it §16):
a 160M–500M training run across **≥4 heterogeneous consumer GPUs** (mixed vendors incl. one
ROCm/Vulkan peer) over WAN, with forced churn, matching the centralized baseline within an agreed ε,
and **round overhead <15%** (incl. the §6.4 barrier-mode ingest gap).

Authored by lane **C3** (Wave 3). Every command below was **exercised** during gate prep; the peer
prep + heterogeneity rehearsal + overhead measurement are all backed by real evidence recorded in
[swarm-ledger-p2-c3.md](swarm-ledger-p2-c3.md). Read that ledger for the "why"; this runbook is the
"how".

> **Non-negotiable resource discipline** (superproject AGENTS.md): one build at a time on this host;
> cap jobs at ≤ nproc/2; never `lint-all`; never bare `-j`. The sealed Windows/CUDA cross-builds run
> at most once. Remote on-box builds (M4/RunPod/Windows) use the remote box's resources and may run
> concurrently with local doc work, but not with a second local build.

---

## 0. The fleet (pinned; verified reachable during gate prep)

| Peer | Access | Backend | Det lane | Role |
|---|---|---|---|---|
| **Strix Halo** (this box) | local | AMD RADV/Vulkan, 128 GB UMA | CPU fp32 | the required **ROCm/Vulkan** peer; run authority + harness driver |
| **Windows 5090** | `ssh usergpu356@37.230.134.194` (cmd.exe) | NVIDIA 5090, DX12+Vulkan 1.4.341, driver 610.74 | CPU fp32 | Windows CUDA/Vulkan peer |
| **RunPod 4090** | `ssh -p 13988 root@213.173.109.230` | NVIDIA 4090, CUDA 12.4 driver 550.127.05 | CPU fp32 | Linux **CUDA** peer |
| **M4 Mac** | `ssh m1@62.210.193.129` | Apple **Metal**, 32 GB unified | CPU fp32 | macOS/Metal peer |
| **M1 mini** | `ssh m1@51.159.120.241` | Apple Metal | — | hosts the live **iroh relay** `http://51.159.120.241:3340` (do not break it) |

**Consensus bar = the det lane, which is CPU fp32 by contract (spec §5.6).** A cross-compiled worker
on any platform must produce **byte-identical** per-round det digests. GPU-lane (wgpu/DX12/CUDA)
native-loss divergence is *allowed and recorded*; det-lane equality is the gate.

**Live substrate:**
- coordinator `https://daemon-swarm-dev.me-dc6.workers.dev` (registry base `…/api/v1/swarm`; wss
  `…/runs/:id/ws`; object-proxy presign R2 plane; internal-identity headers
  `x-daemon-org-id`/`x-daemon-actor` on workers.dev — no gateway).
- relay `http://51.159.120.241:3340` (M1 mini). WS-only runs omit it; dual-plane runs pass it.

**KEEP-IN-SYNC rule:** the deployed coordinator MUST track the merged daemon-cloud coordinator
branch. After any change to `coordinator-wasm`/`registry.ts`/`shell.ts` on `swarm/deploy-dev`,
redeploy via `/tmp/swarm-deploy2` or a fresh scratch clone of
`daemon-cloud/daemon-api` branch `swarm/deploy-dev` (`deploy-dev.sh` render + `wrangler deploy`;
reuse the existing KV/R2/HMAC secret — no rotation). Never touch that repo's working tree.
(Merge-2 caught a stale-coordinator `global_batch=1` data-partition bug this way.)

Reachability preflight (all must return the shown code):
```bash
curl -s -o /dev/null -w '%{http_code}\n' https://daemon-swarm-dev.me-dc6.workers.dev/          # 200
curl -s -o /dev/null -w '%{http_code}\n' http://51.159.120.241:3340/generate_204               # 204
ssh usergpu356@37.230.134.194 "ver"                                                            # Windows banner
ssh -p 13988 root@213.173.109.230 "nvidia-smi --query-gpu=name --format=csv,noheader"          # RTX 4090
ssh m1@62.210.193.129 "uname -a"                                                                # Darwin arm64
```

---

## 1. Build the peer artifacts (sealed, from the merged trunk)

All from the daemon-node worktree at the gate commit, in the devShell (`nix develop --command …`).

**1a. Guests (required before any wasm-backed test / worker attach):**
```bash
cargo run -p xtask -- build-guests           # writes guests/target/.../tiny_llama.wasm + guests.blake3
```

**1b. Linux (this box) worker — the local peer + the harness driver:**
```bash
nix develop --command cargo build -p daemon-train --features swarm-net --bin daemon-train-worker
# -> target/debug/daemon-train-worker   (AMD/RADV box; det lane = CPU fp32)
```

**1c. Windows 5090 worker (sealed MinGW cross-build — NEVER build on-box):**
```bash
nix build .#daemon-train-worker-windows --max-jobs 1 --cores 16 --out-link result-train-worker-windows
scp "$(readlink -f result-train-worker-windows/bin/daemon-train-worker.exe)" \
    usergpu356@37.230.134.194:daemon-train-worker.exe
scp guests/target/wasm32-unknown-unknown/release/tiny_llama.wasm \
    usergpu356@37.230.134.194:tiny_llama.wasm
```
This lane is unblocked by the Wave-3 `daemon-telemetry` fix (the `sentry-rust-minidump` →
`crash-handler` `_invoke_watson` MinGW link blocker is now cfg-compiled out for
`x86_64-pc-windows-gnu`). Verify the deployed worker runs on the real box:
```bash
ssh usergpu356@37.230.134.194 "set DAEMON_TRAIN_PROBE=1 && daemon-train-worker.exe"   # prints Hardware + DeviceLimits
```

**1d. RunPod 4090 CUDA worker (on-box build; the CUDA lane):**
```bash
# From this box: sync the trunk source (excludes target/.git), then build on the pod.
rsync -az --delete --exclude target --exclude 'guests/target' --exclude .git --exclude 'result*' \
    -e "ssh -p 13988" ./ root@213.173.109.230:/root/daemon-node-c3/
ssh -p 13988 root@213.173.109.230 \
  "cd /root/daemon-node-c3 && nix develop --command \
     cargo build -p daemon-train --features swarm-net,cuda --bin daemon-train-worker -j 8"
scp -P 13988 guests/target/wasm32-unknown-unknown/release/tiny_llama.wasm root@213.173.109.230:/root/tiny_llama.wasm
# -> /root/daemon-node-c3/target/debug/daemon-train-worker
```
CUDA runtime note (adjudication (c), the honest nvrtc shape): the pod's driver is CUDA **12.4**;
nixpkgs-unstable's nvrtc is ≥12.6 (its PTX is rejected with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).
The `.#cuda-train` devShell supplies build-time cudart headers; the **runtime** driver-matched nvrtc
12.4 (NVIDIA pip wheel `nvidia-cuda-nvrtc-cu12==12.4.127`) + host driver libs are the one
operator-staged impure input, at `/root/cuda-rt-124`, wired via `DAEMON_CUDA_RUNTIME_DIR`. **There is
no `BackendKind::Cuda` engine arm yet** — `--features cuda` compiles the dep tree and the worker runs
the **CPU det lane** (the consensus bar), so it byte-matches by construction; nvrtc is needed only to
construct an actual burn-cuda backend (C2's `(t+t).sum()=12` smoke).

**1e. M4 Metal worker (on-box build; nix needs a login PATH over ssh):**
```bash
rsync -az --delete --exclude target --exclude 'guests/target' --exclude .git --exclude 'result*' \
    -e ssh ./ m1@62.210.193.129:~/daemon-node-c3/
ssh m1@62.210.193.129 "cd ~/daemon-node-c3 && \
  export PATH=/nix/var/nix/profiles/default/bin:\$HOME/.nix-profile/bin:\$PATH && \
  nix develop --command cargo build -p daemon-train --features swarm-net --bin daemon-train-worker -j 6"
scp guests/target/wasm32-unknown-unknown/release/tiny_llama.wasm m1@62.210.193.129:~/tiny_llama.wasm
# Launch wrapper (nix needs the profile PATH; the worker needs the devShell for its dylibs):
ssh m1@62.210.193.129 "cat > ~/run-worker.sh <<'EOF'
#!/bin/sh
export PATH=/nix/var/nix/profiles/default/bin:\$HOME/.nix-profile/bin:\$PATH
cd \$HOME/daemon-node-c3
exec env DAEMON_TRAIN_MODULE=\$HOME/tiny_llama.wasm nix develop --command ./target/debug/daemon-train-worker
EOF
chmod +x ~/run-worker.sh"
ssh m1@62.210.193.129 "sh ~/run-worker.sh" </dev/null | xxd | head -1   # expect a clean CBOR Ready frame (4f03 0000 a165 5265 6164 79 …)
```

**Per-platform DeviceLimits (probe cross-check, from `DAEMON_TRAIN_PROBE=1`):**
| Peer | vram_mb | ram_mb | shared_mb | unified | lanes |
|---|---:|---:|---:|---|---|
| Windows 5090 | 32190 | 130958 | 0 | false | dx12, vulkan, cpu |
| RunPod 4090 | 0 (cpu-lane) | 127935 | 0 | false | cpu |
| M4 | 25559 | 32768 | 32768 | true | metal, cpu |
| Strix Halo | 4096 (+120000 gtt) | — | — | true | vulkan, cpu |

---

## 2. The heterogeneity harness (the peer driver)

Peers are driven by `TrainSupervisor` over the frozen length-framed CBOR stdio worker protocol. The
`fleet_live_hetero` harness (`tests/daemon-swarm-e2e/tests/fleet_live_hetero.rs`, C3) makes **each
peer's spawn command configurable**, so a peer is a local binary OR a remote process reached over
`ssh` (the CBOR stdio pipes binary-clean over `ssh -T`; validated on Windows/RunPod/M4). It creates
the run, joins every peer, collects per-round det digests, and asserts **byte-identical digests
across platforms every round**.

Peer spec (`SWARM_FLEET_PEERS`, `;;`-separated; each `label|program|arg0|arg1|…`; program `LOCAL` =
the built local worker + module env). The exact 4-peer gate spec:
```
linux-amd-vulkan|LOCAL
;;win-5090|ssh|-T|usergpu356@37.230.134.194|set DAEMON_TRAIN_MODULE=C:\Users\Administrator\tiny_llama.wasm && daemon-train-worker.exe
;;runpod-4090-cuda|ssh|-p|13988|-T|root@213.173.109.230|DAEMON_TRAIN_MODULE=/root/tiny_llama.wasm /root/daemon-node-c3/target/debug/daemon-train-worker
;;m4-metal|ssh|-T|m1@62.210.193.129|sh ~/run-worker.sh
```

Run it (WS-only; add `SWARM_FLEET_RELAY_URL=http://51.159.120.241:3340` for dual-plane):
```bash
SWARM_FLEET_WS_URL='https://daemon-swarm-dev.me-dc6.workers.dev/api/v1/swarm' \
SWARM_FLEET_ROUNDS=6 SWARM_FLEET_WARMUP_S=30 SWARM_FLEET_ROUND_TIMEOUT_S=120 SWARM_FLEET_COOLDOWN_S=3 \
SWARM_FLEET_PEERS='…the 4-peer spec above…' \
  nix develop --command cargo test -p daemon-swarm-e2e --features iroh \
    --test fleet_live_hetero fleet_heterogeneous_det_lane_agrees -- --nocapture --test-threads 1
```

**Gate-prep evidence (all GREEN, real Cloudflare coordinator + real R2):**
- 2-peer Linux+Windows: 6 rounds byte-identical (round 0 `92d31a7d…` … round 5 `bbf40ead…`).
- 3-peer Linux+Windows+RunPod-CUDA: 6 rounds byte-identical (round 0 `82d4f93d…` … round 5 `f6cc48a8…`).
- 2-peer Linux+M4-Metal: 4 rounds byte-identical (round 0 `92d31a7d…` — same as Linux+Windows,
  proving the det lane is truly platform-independent).
- **All four platforms byte-match Linux** (pairwise + 3-way). 3 GPU vendors (AMD, NVIDIA, Apple),
  3 OSes (Linux, Windows, macOS).

**Known gate-prep issue (the 4-peer simultaneous run — READ THIS):** the lean `fleet_live_hetero`
harness (no rejoin) stalls at round 0 with all four WAN peers **when the M4 peer is spawned via
`nix develop --command` over ssh** (its per-spawn latency + `min_peers=N` barrier + no rejoin ⇒ an
early park that does not self-heal). Isolation runs proved: the det lane and links are fine (Linux+M4
2-peer green; a **local** 4-peer run green — see §4). Mitigations for the ceremony, in order:
1. **Prefer the churn-robust driver.** Drive the 4-peer gate run with the Merge-2
   `ws_live_workers.rs` harness (it has drop→park→rejoin recovery), extended with the same remote-ssh
   peer specs — a park then self-heals. This is the recommended ceremony driver.
2. **Pre-warm M4.** Replace `run-worker.sh`'s `nix develop --command` with a pre-realized profile:
   `nix develop --command true` once to warm, or bake a `nix print-dev-env > ~/dev-env.sh` and
   `source ~/dev-env.sh; exec ./target/debug/daemon-train-worker` so spawn is instant.
3. **Generous timings** (`SWARM_FLEET_ROUND_TIMEOUT_S=120`) — necessary but not sufficient alone.

---

## 3. Run creation (envelope / RunConfig)

The run author freezes the envelope (§6.1) and declares the RunConfig on the create request (Merge-1
Decision 1 — the registry never parses the envelope). The harness does this for you; the shape (for a
hand-authored gate run) is:

- **Envelope** (`author_envelope`): `min_peers = max_peers = N` (adjudication (d): a join after the
  `WaitingForMembers`→Warmup transition is staged pending and, with `epoch_rounds=0`, never
  materializes mid-run — so set `min_peers` to the exact initial roster), `round_mode=barrier`,
  `payload_store="r2"`, `global_batch = N × steps_per_round × micro_batch` (must divide evenly across
  the roster), `stop=Rounds(R)`, the tiny-llama (rehearsal) or 160M (gate) module in `[artifacts]`.
- **CreateRunRequest** (declared RunConfig): `warmup_timeout_s`, `round_timeout_s`, `cooldown_s`,
  `global_batch`, `witness_target`, `update_max_bytes`, `min/max_peers`, `rounds` — forwarded verbatim
  to the DO `init`. Size `round_timeout_s` to comfortably exceed the slowest peer's per-round wall.

For the **160M gate model**, swap the guest for the 160M preset envelope + the tokenized corpus
(`cargo run -p xtask -- tokenize-corpus …`), and raise `steps_per_round` so each round is seconds of
real compute (this is what makes the <15% overhead figure gate-representative — see §5).

---

## 4. Observe / replay instrumentation

Every gate run MUST be captured so a failure replays offline (PROTO-20 oracle). Two capture paths:

**4a. The `swarm-local` transport harness (loopback/iroh, StubBackend — fast, GPU-free):**
```bash
nix develop --command cargo run -p daemon-swarm-run --features iroh --bin swarm-local -- \
  --transport iroh --store fs --peers 3 --rounds 8 --relay http://51.159.120.241:3340 \
  --observe ./gate-capture
nix develop --command cargo run -p daemon-swarm-run --features iroh --bin swarm-replay -- ./gate-capture
```
Merge-2 verified: a live iroh-mesh run over the M1 WAN relay, 8/8 rounds re-derived **byte-identically**
by `swarm-replay` (`committed=3 attested=3 finalized=true digest_agreed=true` every round).

**4b. The worker-subprocess loop (the real heterogeneous peers).** The observe surface currently
rides the `swarm-local`/`live_transport` harness path; wiring `--observe` into the cloud-DO
worker-subprocess loop (`ws_live_workers`/`fleet_live_hetero`) is a small carried follow-on (Merge-2
Task-5 note; Wave-3 launch note). Until then, the cross-peer consensus record for a worker-subprocess
gate run is the **per-round digest transcript** the harness prints (byte-identity across peers is the
same assertion the replay oracle makes), plus the coordinator's `RoundRecord` objects in R2 (the
authoritative, replayable log). For the archived offline-replay artifact, run a parallel `swarm-local
--observe` iroh mesh alongside the worker run, or land the worker-loop `--observe` follow-on first.

---

## 5. ε-convergence + overhead (the two numeric gate criteria)

### 5a. Overhead (<15%, incl. the §6.4 barrier ingest gap)

**Definition:** `overhead% = (mean N-peer round wall − single-host baseline round wall) / baseline`.

**Tool (C3):** `fleet_live_hetero::swarm_round_overhead_vs_single_host` runs the SAME workload as a
1-peer baseline and an N-peer swarm (all local subprocesses, real guest + real R2) and prints the
figure:
```bash
SWARM_FLEET_WS_URL='https://daemon-swarm-dev.me-dc6.workers.dev/api/v1/swarm' \
SWARM_FLEET_ROUNDS=8 SWARM_FLEET_OVERHEAD_PEERS=4 \
  nix develop --command cargo test -p daemon-swarm-e2e --features iroh \
    --test fleet_live_hetero swarm_round_overhead_vs_single_host -- --nocapture --test-threads 1
```
**Rehearsal numbers (tiny-llama, local 4-peer, this box → dev coordinator):** single-host
`t1 = 1051 ms`, 4-peer `tN = 1219 ms` ⇒ **cross-peer overhead +168 ms/round = 16.0%**.

**Read this correctly:** at tiny-llama scale the per-round *compute* is sub-millisecond, so the round
wall is dominated by the ~1 s R2/WS WAN round-trip and the 16% is protocol-latency-dominated — it is
**NOT** the gate figure, only mechanism validation + the barrier's absolute cost (~168 ms of
cross-peer fetch/aggregate). At the **160M gate model** each round is *seconds* of real compute, so
the same absolute barrier is a small fraction: run the exact tool above with the 160M envelope and
the reported overhead is the gate number. Cross-check against the centralized single-host throughput
baseline (§5b).

### 5b. Centralized baseline + ε-convergence

1. **Centralized 160M baseline:** the reference-parity throughput harness on one host
   (`swarm-p2-throughput.md` method) gives the single-host loss curve + tokens/s:
   ```bash
   M2_WGPU_WARMUP=3 M2_WGPU_MEASURED=10 nix develop .#vulkan --command \
     cargo test -p daemon-train --features wgpu --release --test reference_parity_wgpu \
     throughput_within_budget_or_documented -- --ignored --nocapture --test-threads=1
   ```
   (160M wgpu/RADV baseline on this box: ~384 tok/s lazy-backend; loss curve 10.85→4.93 over the
   reference steps — the ε reference.)
2. **Swarm run:** the full WAN 160M run (§2 + §3, 160M envelope). Capture the per-round aggregated
   loss (the `Metric` events / observe capture).
3. **ε criterion:** the swarm loss curve tracks the centralized baseline within the agreed ε over N
   epochs, with **zero det-digest mismatches** (§2 assertion) and the replay oracle green (§4).

---

## 6. Churn drill (forced, mid-run — per the §6.4 stall ladder)

Churn is churn-normal, not an error. The Merge-2 `ws_live_workers.rs` harness encodes the canonical
drill and is the recommended ceremony driver:
- **Kill** a peer after it reports round K (mid-run kill) — genuine churn (a distinct roster identity
  leaves).
- The coordinator ages it out via **K record-absences** (§6.4 rung 1) → floor breach (`min_peers`) →
  the run **parks** in `WaitingForMembers`.
- Node-side supervision **respawns** the child (lazy respawn, §10.3/§13); it **re-assesses** and
  **rejoins** (a previously-Dropped member may rejoin, §6.5); warmup re-runs; the run resumes and
  **finishes**.
- **Stall drill** (§6.4 rung 2): delay one payload past the barrier → the peer stalls, skips a
  training round, late-ingests within `payload_retention_rounds`, catches up within
  `stall_rounds_max` (default 2).

Merge-2 verified this end-to-end on the real Cloudflare substrate + M1 WAN relay: 8 rounds, mid-run
kill of a peer, drop→park→respawn→rejoin→finish, survivors byte-identical throughout (run ids
`run-a3-e2e-1783995994` WS-only and `…-1783996154` dual-plane).

---

## 7. Pass / fail criteria (spec §17)

The gate PASSES iff **all** hold over the full run:
1. **≥4 heterogeneous peers**, mixed GPU vendors incl. ≥1 ROCm/Vulkan peer, over WAN — the fleet
   above satisfies this (AMD/RADV + NVIDIA/Windows + NVIDIA/CUDA + Apple/Metal).
2. **Zero det-digest mismatches** — every peer that reports a round agrees byte-for-byte (§2).
3. **ε-convergence** — swarm loss tracks the centralized 160M baseline within the agreed ε (§5b).
4. **Round overhead <15%** incl. the barrier ingest gap, measured at 160M scale (§5a).
5. **Forced churn survived** — kill/rejoin + stall drills complete; the run reaches Finished (§6).
6. **Replay oracle green** over the full run log (§4).

Any det-digest mismatch is a **hard fail** (consensus broken) — capture, then replay offline to
localize. ε or overhead outside bound is a research finding, recorded with the numbers.

---

## 8. CI tiers (the standing gate around the ceremony) — TDD §8.1

Three layers, because bit-identity is a CPU property (the det lane never runs on GPU), so CI needs
GPUs only at tiers 2/3:

- **Tier 1 — per-PR, hosted CI, no GPU (implemented).** The consensus-critical CPU suites: shared
  det kernels, round protocol + codecs, harness/assess/replay, observe oracle, worker det lane +
  cross-backend digest identity + wasm-guest determinism, SDK goldens, e2e drills, wire conformance.
  Single in-repo definition: **`cargo run -p xtask -- swarm-ci-det`** (builds guests, then the pinned
  suite list, red-fails on the first). The superproject CI job is a thin caller (§8a). GPU + live
  lanes are excluded here by design.
- **Tier 2 — per-lane scheduled, self-hosted runners (design; runbook, not fake CI YAML).** One box
  per backend (CUDA / Vulkan-RADV / Metal), scheduled (nightly/weekly), each running the native-lane
  tolerance-class fixtures + HOST-9 parity + the cross-lane replay (native losses within tolerance,
  det digests byte-equal). The lane commands are exactly the on-box builds + suites in §1c–1e (e.g.
  `cargo test -p daemon-train --features wgpu` on the RADV box; `--features cuda,burn-ndarray` on the
  CUDA box; the Metal suite on M4). One runner per lane suffices — cross-peer identity never depends
  on the GPU. These boxes are not CI-reachable, so this is an operator/homelab schedule, not a
  workflow file for machines CI can't reach.
- **Tier 3 — hardware-in-loop, manual (this runbook).** The ≥4-peer mixed-vendor WAN ceremony, per
  release candidate. Not CI: it runs by hand with pinned envelopes and archived `RoundRecord` logs.

### 8a. Proposed superproject CI job (LEDGER/PROPOSAL — the `.github/workflows/ci.yml` is the human's, signed)

daemon-node has no CI of its own; CI lives in the superproject `.github/workflows/ci.yml`. Its
`daemon-node` job already runs `cargo test --workspace` (which covers the default-feature swarm
suites). Add an **additive** job that runs the tier-1 definition (which also exercises the
feature-gated CPU suites `swarm --workspace` does not, e.g. `daemon-train --features burn-ndarray`,
`daemon-train-sdk --features sim`):
```yaml
  swarm-det:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - uses: DeterminateSystems/nix-installer-action@main
      - uses: cachix/cachix-action@v15
        with: { name: daemon-ai, authToken: ${{ secrets.CACHIX_AUTH_TOKEN }} }
      - name: swarm CI tier-1 (CPU consensus-critical det/protocol/codec/wasm suites)
        run: cd daemon-node && nix develop --command cargo run -p xtask -- swarm-ci-det
```
This is a proposal for the human to apply + sign; do not edit the main-repo working tree from a lane.

---

## 9. Rollback / cleanup

- **Runs** are ephemeral on the coordinator (each `run-*` id is unique; the registry rejects
  duplicates). No teardown needed; stop the peers with the harness `Leave`+`Shutdown` (the harness
  does this in its cleanup) or Ctrl-C.
- **Fleet boxes (good-guest):** all artifacts live in temp/home dirs and are deletable —
  Windows `%USERPROFILE%\daemon-train-worker.exe` + `tiny_llama.wasm`; RunPod `/root/daemon-node-c3`
  (+ `/root/cuda-rt-124`, `/root/tiny_llama.wasm`); M4 `~/daemon-node-c3` + `~/tiny_llama.wasm` +
  `~/run-worker.sh`. The M1 relay is a live service — **do not touch it**.
- **Local build artifacts:** reclaim finished lane `target/` with
  `daemon-worktree/clean-lane-target.sh <worktree-dir>` (refuses on uncommitted tracked changes / an
  active build).
- **Coordinator:** never left in a bad state by a run; if a redeploy is needed, re-run `deploy-dev.sh`
  (KEEP-IN-SYNC rule, §0). Never touch the daemon-cloud working tree.

---

## 10. One-screen checklist

1. Preflight reachability (§0); redeploy the coordinator if the cloud branch moved.
2. `build-guests`; build + deploy the four peer workers (§1).
3. Probe each peer (`DAEMON_TRAIN_PROBE=1`) — confirm DeviceLimits (§1).
4. Author the 160M envelope + declared RunConfig; create the run (§3).
5. Start observe capture (§4).
6. Join all ≥4 peers (churn-robust driver — §2 + §6); run to Finished.
7. Assert: zero digest mismatches (§2/§7), ε-convergence vs baseline (§5b), overhead <15% (§5a),
   churn survived (§6), replay green (§4).
8. Record numbers + digests in the ledger; clean up (§9).
