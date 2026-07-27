# VHC program state

**This is the only program document an agent reads by default.** It is rewritten — never
appended — at every program boundary, and it is status, not normative text. Normative text lives
in the tracked specs (§2). If this file contradicts a chat log, a memory, or an archived
document, this file wins; if it contradicts a tracked spec, the spec wins.

Last rewritten: 2026-07-27 (program reset).

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
  through the product path: training rounds, digest agreement, churn/hard-kill restore, replay.
  This is the existing acceptance lane, pinned as the merge gate's core. Status: lane exists
  (`tests/daemon-vhc-acceptance`); promotion to pinned C0 rung in progress.
- **C1 — on demand / nightly.** Two real boxes, real transport (relay), small geometry; adds
  WAN churn drills and remote product-path drive. Status: not yet run under this framing.
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

- Integration trunk: `vhc-integration` @ `b895d7ee` (resource subsystem + host wave merged).
- Admitted-tuple extension (session-side tuple + tests, ~840 lines): staged work recovered from
  the stopped workers; being verified and committed as part of trunk convergence.
- Guest line: `vhc/guest-wave-a` @ `5f8e1f4b`, green on its own gates; merge into
  `vhc-integration` pending (trunk convergence).
- Doctrine amendment (§3) to the tracked specs + code: pending.
- C0 promotion: pending. C1: not run. Fit probes: not run. Freeze/C2: not reached.
- Program archive: frozen and locked read-only 2026-07-27.

## 8. Next actions (in order)

1. Land the doctrine amendment: spec text (§3) + estimate/verdict types; delete the caller-less
   selection path; rationalize the plan/claim/profile vocabulary.
2. Converge the trunk: commit the admitted-tuple work, merge the guest line, prune topic
   branches, one worktree per line.
3. Promote the acceptance lane to the pinned C0 rung of the mandatory gate.
4. Run C1 on two boxes; fix what it surfaces.
5. Fit probes on all three boxes at ceremony geometry (one fixed retention policy).
6. Freeze; memoized preflight; run C2; evidence closure; human-signed master merge.

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
