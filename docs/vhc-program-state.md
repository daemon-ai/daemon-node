# VHC program state

**This is the only program document an agent reads by default.** It is rewritten — never
appended — at every program boundary, and it is status, not normative text. Normative text lives
in the tracked specs (§2). If this file contradicts a chat log, a memory, or an archived
document, this file wins; if it contradicts a tracked spec, the spec wins.

Last rewritten: 2026-07-27 (C1 transport delta landed as a gate: relay-carried mesh + churn in
the acceptance lane; gapped-restore divergence fixed — `StaleRestore` outcome + coordinator
replay-forward; operator provisioning command shipped; physical two-box run blocked on box
reachability).

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
- C0: green and pinned. C1: software delta green; two-box run blocked on hardware (§8.1).
  Fit probes: not run. Freeze/C2: not reached.
- Program archive: frozen and locked read-only 2026-07-27.

## 8. Next actions (in order)

1. **[HUMAN] Restore fleet reachability.** The relay box (`intelligent-city-fades-fin-03`,
   31.22.104.86) times out on ssh, and the Mac / Windows seats are not in the build host's ssh
   config. C1's physical run needs one reachable second box; C2 needs all of them.
2. Run C1 on two boxes: provision each box with `vhc-provision-dev-profile`, start the relay
   per the runbook §4.2, point both nodes' `[vhc.iroh]` at it with WAN `advertise_ips`, drive
   over `ssh → daemon-cli`; fix what it surfaces. The transport/churn semantics are already
   gate-proven; this run is about real WAN, real hardware heterogeneity.
3. Fit probes on all three boxes at ceremony geometry (one fixed retention policy); the probe
   runner records content-addressed verdicts (`[RC-15]`).
4. Freeze; memoized preflight; run C2; evidence closure; human-signed master merge.

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
