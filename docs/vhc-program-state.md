# VHC program state

**This is the only program document an agent reads by default.** It is rewritten — never
appended — at every program boundary, and it is status, not normative text. Normative text lives
in the tracked specs (§2). If this file contradicts a chat log, a memory, or an archived
document, this file wins; if it contradicts a tracked spec, the spec wins.

Last rewritten: 2026-07-27 (fit-probe machinery landed as gates and run for real: the build
box holds a GREEN ceremony-geometry FitVerdict on its Vulkan lane; two device-lane defects the
probe surfaced are fixed — WGSL bool-storage refusal (SPIR-V compile lane) and the epoch
watchdog epoch-killing a live device slice (liveness extension, ABI §5.6); remote-box probes
blocked on the same reachability as C1).

## 1. What we are building, and the next milestone

daemon-vhc is a virtual heterogeneous cluster: every unit of run policy — the training algorithm
and the consensus protocol — is a sandboxed wasm module authored against the SDK, and the host is
capabilities plus cryptographic mechanism (four worlds: `compute@` Burn-shaped, `net@`, `data@`,
`sys@`). Modules train on the native lane and reconcile on the bit-exact det lane.

The next milestone is **C2, the fleet ceremony** (§5): a **replicated-training protocol proof on
heterogeneous consumer hardware** — a TinyLlama-class model trained by three trainer boxes over
real WAN through the product path only, with zero det-digest mismatches, churn drills, checkpoint
restore, and a green offline replay (criteria G-1..G-6 in the ceremony runbook). It is *not* a
7B/40B product certification, *not* a performance study, and *not* a measurement campaign.
Sharded/offloaded execution at larger scales is post-ceremony work.

## 2. Canonical documents

- `docs/specs/vhc-architecture-spec.md` — architecture + resource model (normative)
- `docs/specs/vhc-module-abi-spec.md` — module ABI (normative; drift-gated against the code)
- `docs/specs/vhc-fleet-ceremony-runbook.md` — ceremony definition and runbook (normative)
- `docs/vhc-program-state.md` — this file (status)
- Program history: `~/experiments/decentralised-llm-training/archive/` — **frozen, read-only,
  off-limits to agents**. Never cite it as authority; it is human archaeology only.

## 3. Resource doctrine — the machine is the oracle

The three resource artifacts keep their owners: the guest derives a **logical** resource plan
(no device physics), the backend implementation owns its **execution profile**, the node owns
its measured **capability report**. What changed at the reset is their epistemic role:

1. **Composition output is a conservative estimate, not an authority.** It exists for exactly
   two purposes: sound cheap refusal (if even the estimate exceeds supply, refuse without
   probing) and sizing the enforced runtime budget. Exact high-water prediction is a non-goal.
2. **The fit probe is the authority.** Feasibility of (module, backend implementation revision,
   geometry, grant) is established by running the real module on the real backend at the real
   geometry under the enforced budget, and recording the outcome as a content-addressed,
   memoized **fit verdict**. Fleet feasibility for a frozen roster = every node holds a green
   verdict. There is no roster-wide search or matching protocol while membership is frozen.
3. **A wrong estimate is a contained, typed runtime failure — never an outage and never a human
   question.** Byte, margin, page-size, and allocator questions are not escalatable to the
   owner; the probe answers them. Only product-semantic questions escalate (§6).

## 4. Rules are gates

A rule that is not enforced by a named mechanical gate (test, lint lane, drift check, type) is
advisory. The gate battery is `cargo run -p xtask -- vhc-production-gate` (plus the det/node/t2
lanes and the acceptance suite); the codename scan and the ABI drift check stay mandatory. The
rule-identifier namespace is frozen: no new rule families, and an amendment to the tracked specs
must delete or subsume at least as much normative text as it adds.

## 5. Ceremony ladder

Each rung is a strict subset of the one above and runs continuously below it. Defects must be
discovered at the lowest rung capable of showing them.

- **C0 — every merge (CI).** One box, three real daemon processes (coordinator seat + trainers)
  through the product path: training rounds, digest agreement, churn/hard-kill restore, replay,
  live module switch. This is the acceptance lane (`tests/daemon-vhc-acceptance`), pinned as the
  merge gate's non-negotiable core (`xtask vhc-acceptance`, folded into `vhc-production-gate`).
  Status: **green** — the plan-emitting trainer admits through PC-12 dev-authority provisioning;
  all twelve gates pass.
- **C1 — on demand / nightly.** Two real boxes, real transport (relay), small geometry; adds
  WAN churn drills and remote product-path drive. Status: **software delta green, hardware
  blocked.** The relay-carried transport posture is a merge gate now
  (`iroh_relay_plane.rs`: real `iroh-relay`, roster records with zero direct addresses so the
  relay is the only dial path, training + the full graceful-churn choreography through it).
  The physical two-box execution is blocked: the relay box times out and no other fleet seat
  is reachable from the build host (§8.1).
- **C2 — the milestone.** Three boxes over WAN at ceremony geometry per the runbook
  (G-1..G-6), on a frozen candidate. Preflight is only what G-1..G-6 require — relay, sealed
  binaries, identities, corpus, genesis, seat, and a green fit verdict per box — each item
  memoized by content hash so a re-freeze re-runs only what changed. Per-OS conformance and
  profile-calibration campaigns are **post-ceremony** product work, not preflight.

## 6. Open owner decisions (maximum five; product semantics only)

1. **Milestone labeling** — C2 is claimed as a replicated-training protocol proof on the frozen
   three-box roster (not a product certification). Confirm at freeze.
2. **Availability verification in the ceremony genesis** — on (full coverage) vs off (recorded
   as reduced coverage). Decide at freeze.
3. **Numerics bar** — the tolerance class accepted for trained state across backends. Confirm
   the ratified optimizer-class tolerance at freeze.
4. **Privacy/threat posture** — corpus, journals, and metrics are egress channels; confirm the
   accepted posture for the public ceremony evidence.

None of these blocks C0 or C1.

## 7. Current state (rewrite this section at every boundary)

- Integration trunk: `vhc-integration`, converged — admitted-tuple extension committed, guest
  wave merged green (`98eae678`), topic branches pruned, one worktree per line.
- Doctrine amendment: landed — `[RC-15]` (estimate demoted, verdict is authority) in the
  architecture spec; composed-side `claim` vocabulary renamed to `estimate` across specs, code
  and wire keys; `FitVerdict`/`FitProbeKey`/`FitVerdictStore` types in `daemon-vhc-resource`;
  the caller-less estimate-driven selection path (`select`/`SelectionPolicy`/`validate_against`)
  deleted.
- PC-12 provisioning: landed. The provisioned-profile file (`daemon-vhc-resource::provision`,
  `DAEMON_VHC_PROFILE_DIR`, path-reference like the identity store) carries profiles + envelopes,
  the owner acceptance policy, and lane bounds; the node hands `<data>/vhc/profiles` to workers;
  the worker assembles a `ResourceAuthority` from it at every admission site (assess, join
  re-verify, switch fence). Development authorities are a separate, doubly-opt-in acceptance
  (`accepted_development_authorities` on BOTH the owner policy and the run's requirements);
  integration evidence only, never ceremony certification. The worker's
  `DAEMON_TRAIN_REVISION_OUT` mode exports its own revision records for provisioning tools.
  An un-provisioned box still refuses `EstimateNotComposable`, typed (asserted by test).
- Genesis authoring: requirement derivation runs the module's assessment against the SAME
  plan-relevant config the genesis pins for the role (empty-config assessment was refused by the
  plan-emitting trainer, rightly). A CPU-lane capability report states `NotApplicableToLane` for
  the per-allocation ceiling; the pool-bound check validates a CPU estimate against usable supply.
- The node's join surfaces an ineligible assessment's own reasons instead of converting every
  assess refusal into a "nothing to reserve" internal error.
- Operator provisioning: `cargo run -p xtask -- vhc-provision-dev-profile --worker-bin <bin>
  --out <dir> --authority <64-hex> [--class cpu]` runs the box's own worker in revision-export
  mode and writes the PC-12 provisioned file the node's `DAEMON_VHC_PROFILE_DIR` reads. xtask is
  the ONE named non-dev edge `vhc-dep-check` permits to enable the profile-minting
  `test-support` feature (dev tooling, never shipped; minting is not vouching — acceptance
  still requires the owner policy and the run's genesis to both name the authority).
- C1 relay gate: landed in the acceptance lane (`iroh_relay_plane.rs`, ~160 s, rides
  `xtask vhc-acceptance`). Relay-only reachability is representable in shipped config:
  `[vhc.iroh] relays = "<url>"` + `advertise_ips = []`.
- Gapped-restore divergence (found by the relay gate under load): a rejoiner whose restored
  watermark lagged the live round silently skipped committed rounds and forked the det
  trajectory. Fixed at both ends, both gate-proven: the rounds SDK refuses a gapped fold
  (`Outbound::GapRefused` → ABI §4.5 outcome 3 `StaleRestore`, classified RETRYABLE — the
  only nonzero outcome that is, because a retry restores a fresher checkpoint), and the
  coordinator's join admission re-publishes its retained ring of committed records ascending
  (replay-forward), so a rejoiner inside the retention window folds the gap instead of ending.
- Fit probes (`[RC-15]`): the runner exists end to end and has produced its first real verdict.
  The worker's `DAEMON_TRAIN_FIT_PROBE` mode drives the actual module on the actual measured
  backend at the granted geometry under the enforced budget, consuming only pre-authored opaque
  inputs (`daemon-vhc-resource::probe` directory contract — the worker binary links no round
  vocabulary; authoring lives in `daemon-vhc-testkit::fit_probe` / `xtask vhc-fit-probe`), and
  records a content-addressed FitVerdict keyed by (module, backend revision, plan, grant,
  budget). Gate: `tests/fit_probe.rs` in the worker crate (CPU lane, green).
- **This box holds a GREEN ceremony-geometry verdict on its Vulkan lane** (Radeon 8060S/RADV):
  round 0 committed, measured peak 43.3 MB guest linear memory under the 142.2 MB budget.
  Evidence: `~/experiments/ceremony-artifacts/fit-probe-20260728-platform-vulkan-spirv/`
  (the RED predecessors and their logs are retained there — they are the defect evidence).
- Two real device-lane defects the probe surfaced, both fixed and gate-proven:
  1. cubecl's WGSL lane emits `var<storage> array<bool>` buffers (cmp/mask kernels), which WGSL
     forbids as host-shareable — every such kernel was refused on RADV (`ComputeFault`). Fixed
     by compiling the Vulkan adapter's kernels to SPIR-V (`burn/vulkan` beside `burn/wgpu` in
     the host's `wgpu` feature).
  2. The epoch watchdog epoch-killed a LIVE device slice: a ceremony-geometry round is one
     slice whose wall grows with granted geometry (this box: ~690 s of device compute), so no
     per-slice wall constant separates working from wedged. The deadline now extends on expiry
     iff the guest entered a host import since the last expiry (host contact is the liveness
     proof; device wall lives inside imports), and interrupts only a full budget with zero
     import entries — pure-wasm spins still die within two budgets, fuel/op budgets untouched
     (ABI §5.6 amended). Also: wasmtime reports the epoch trap as `interrupt`, which the
     classifier misfiled as `BadModule`; now `BudgetEpoch`.
- C0: green and pinned. C1: software delta green; two-box run blocked on hardware (§8.1).
  Fit probes: green on this box's ceremony lane; M4/Windows probes blocked on the same
  reachability (§8.1). Freeze/C2: not reached.
- Program archive: frozen and locked read-only 2026-07-27.

## 8. Fleet roster and next actions

The roster is status this file MUST carry (an earlier revision dropped it into the archive and
a session then declared reachable boxes unreachable — access facts live here, verified by ssh):

| Box | Access | Hardware | Role | Backend |
|---|---|---|---|---|
| Strix Halo (build host) | local | AMD Strix Halo, 128 GiB UMA, RADV | trainer + coordinator seat + operator seat | wgpu/Vulkan |
| M4 Mac | `ssh m1@62.210.193.129` | Apple M4, 32 GiB unified (memory floor) | trainer | wgpu/Metal |
| Windows 5090 | `ssh usergpu356@37.230.134.194` (cmd.exe; build via sealed Nix cross-build, never on-box) | RTX 5090 32 GiB, Server 2022 | trainer | wgpu/DX12 |
| M1 mini | `ssh m1@51.159.120.241` | Apple M1, 8 GiB | iroh relay only | — |

All four answer ssh (verified 2026-07-28). Next actions (in order):

1. Fit probes on the M4 (Metal) and Windows 5090 (DX12) at ceremony geometry:
   `vhc-provision-dev-profile` then `vhc-fit-probe` per box (on-box for the Mac; authored
   here + driven over ssh for Windows).
2. Run C1 on two boxes: start the relay per the runbook §4.2, point both nodes' `[vhc.iroh]`
   at it with WAN `advertise_ips`, drive over `ssh → daemon-cli`; fix what it surfaces. The
   transport/churn semantics are already gate-proven; this run is about real WAN and real
   hardware heterogeneity.
3. Freeze; memoized preflight; run C2; evidence closure; human-signed master merge.

## 9. Agent contract

- **Read this file and your task. Nothing else by default.** The archive is off-limits; do not
  reconstruct program history; do not re-derive decisions recorded here or in the specs.
- **One line of work on the critical path at a time.** No parallel waves on the same worktree;
  measurement/probe builds pin a committed revision, never a working tree.
- **A deliverable is a gate that runs green**, not a report. Reports are one page or less.
- **Never create a normative document, a rule family, or an append-only ledger.** Spec changes
  are commits to the tracked specs. Run evidence goes into a content-addressed directory under
  `~/experiments/ceremony-artifacts/<run-id>/` with a one-line pointer here — never inlined
  into markdown.
- **Timebox investigations.** If a probe or a test can answer the question, run it instead of
  investigating. Escalate only product-semantic questions (§6), never numbers.
- **Resource discipline is unchanged**: capped jobs, one build at a time, diff-scoped lint.
