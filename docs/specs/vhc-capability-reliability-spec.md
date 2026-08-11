# VHC Capability-Seam Reliability Specification

**Subsystem:** VHC — the environment/consensus boundary (capability providers, trap taxonomy,
node liveness), post-C2 workstream.
**Status:** design specification, **implementation in progress** — clauses flip individually
to **LANDED** with a date as their rungs land (currently: §3 REL-2, §3.1 REL-2a, §4 REL-3,
§5 REL-4, §6 REL-5, §6.1 REL-5a, §9 REL-7 a/b/d, §10 REL-8, §11 REL-9); everything else
remains specification.
Revision 2 (2026-08-10) incorporates an external design review: the object-store 403 taxonomy
(§3), the ABI-minor assignment (§7), the demotion of trap attribution to a bounded heuristic
(§5), and the qualified evidence claims (§3, §8). Revision 3 (same day, second review round)
adds the presign-cache freshness requirement (§3 — "fresh presign per attempt" is not what the
cache currently guarantees), bases the §5 heuristic on the existing slice execution-context
scaffold instead of an independent watermark, and records the existing unknown-outcome
degraded-reading drift (§7). Revision 4 (2026-08-11) adds the records-transport seam (§9,
REL-7) after C2's round-46 sequence-gap incident, names exact-frame archive backfill as the
transport seam's principled future lane (§13), and adds the offline assembler corollary
(§3.1, REL-2a) after C2's closure pulls lost whole 30–50 minute passes to single transients.
Revision 5 (2026-08-11, post-C2-execution review) widens the workstream to the fault classes
C2's full execution actually produced (~a dozen incidents across six classes, several absent
from earlier revisions): the storage lifecycle at the recovery seam (§10, REL-8 — as many
manual interventions as the guest-panic class), keeper stall *reaction* and completion
stand-down (§11, REL-9), grant staleness under checkpoint rotation (§12, REL-10 — a wholly
new defect class), guest-side membership decay and the concurrent-churn headroom rule (§7
additions), and run-vs-session telemetry separation (§6.1, REL-5a). Open design questions are
registered in §16 — this document deliberately records them rather than pretending they are
settled. Revision 6 (2026-08-11, **Rung 0 executed**) records the decoded C2 archive evidence
(§2.1) and threads its corrections through the clauses: the flagship 23:09Z panic is
**reattributed** from R2 egress to buffer-handle quota exhaustion (`GRANT_EXHAUSTED`), the
archived completion codes were NOT uniformly `STORE_REFUSED`, a sixth panic-feeding class
(evicted det-state chunk under a sealed fold, RQ-11) surfaced, and the co-located-trainer
respawn lane's unbounded flap cycle (461 attempt-0 cycles at flat 1 s pace) is pinned to its
counter-reset-on-spawn-success (RQ-6). Revision 7 (2026-08-11, Rungs 1a+1b landed) flips §3
(REL-2: GET-side absorption, the 403/expiry taxonomy, presign freshness) and §4 (REL-3: the
shared `comp_error_code` mapper; RQ-2 resolved inline) to LANDED, both verified by unit tests
only — no ceremony time consumed. Revision 8 (2026-08-11, Rung 2 landed) flips §5 (REL-4: the
slice-context attribution heuristic) to LANDED on the same unit-test-only discipline; RQ-3
(whether the constraints are the right final rule) deliberately stays open. Revision 9
(2026-08-11, Rung 3 landed) flips §6 (REL-5: the stall observer, dual watermarks, the
stateful `run_stalled`/`run_progress_resumed` pair) and §6.1 (REL-5a: honest session phase
strings) to LANDED, records one deviation (the per-run threshold adapts to the *observed*
inter-commit gap over a 10-minute floor, because the authored round wall is not visible
host-side), and narrows RQ-4 to its residual (freezing the defaults against a live
checkpoint-heavy run). Revision 10 (2026-08-11, Rung 1c landed) flips §3.1 (REL-2a: resume
moved into the assembler with `*_reused` reporting; the module-fallback GET joins the retry
contract; the C2 field patch superseded and its redundant outer retry removed) to LANDED,
verified by the testkit assembly regression (network-unplugged resume + tamper-and-repair).
Revision 11 (2026-08-11, Rung 5 landed) flips §9 (REL-7 a/b/d: gap-hold visibility on
`plane_health`, the runbook transport-posture preflight item, and the co-trainer cycle
budget closing the 461-cycle unbounded-respawn defect) to LANDED; RQ-5's packet-loss residual
and RQ-6's lane-separation half deliberately stay open. Revision 12 (2026-08-11, Rung 6
landed) flips §10 (REL-8: reclaim before every re-admission, the disk-floor refusal joining
the storage-gate lane instead of the budget lane, the authoring storage gate, and the
`storage_pressure` pre-kill warning) to LANDED on the unit-test-only discipline; RQ-9 is
narrowed to its defaults-freeze residual (the per-round growth figure still comes from the
operator, not from banked preflight evidence automatically). Revision 13 (2026-08-11, Rung 7
landed) flips §11 (REL-9: the bounded stall-recycle riding the ordinary retryable lane, and
the completion stand-down checking run-terminal evidence before any retry) to LANDED; RQ-8
is decided conservatively — external run-head progress is the ONLY reaction condition taken,
the whole-run-wedge exception is deliberately NOT implemented (revisit with C3's
decay-while-waiting fix, which removes most of that class at the source). Revision 14
(2026-08-11, Rung 8 landed) flips §12 (REL-10: continuous evidence-based grant extension at
the `PayloadPut` seam — the `da_migrate` precedent generalized; the `grants.cddl`
artifacts-field drift repaired beside it) to LANDED on the unit-test-only discipline; RQ-10
(first-class grant classes/planes) stays a C3 ABI-minor design item. Revision 15
(2026-08-11, Rung 4 landed) flips §7 (REL-6: ABI minor 6 assigning `OUTCOME_ENV_STARVED = 4`
→ host `FailedRetryable`; reserved-code degrade-to-`Left` drift repaired in the same commit;
tiny-llama's classified panic-site audit converting the environment-sensitive seams — 2814
payload fetch, 1834 restore window, 1955 round-base window, init manifest/window, container
put, moment export — to typed run-ends with `GRANT_EXHAUSTED` as bounded backpressure; the
8b admitted-on-own-membership fix; decay-while-waiting in the coordinator SDK) to LANDED on
the unit-test-only discipline (consensus-tick two-dead-trainer + window/liveness scenarios,
ABI negotiation tests, session outcome-arm tests; whole-run testkit drills and the G-3 shapes
belong to the next ceremony window, per the no-long-C2-tests constraint). Runbook §4.7
headroom generalized to ≥ expected concurrent churn. Revision 16 (2026-08-11) adds the
desync regression lattice to §9: the session's gap-aging ladder unit-pinned over a new
abstract delivery seam (`retry_held_with` — deadline verdict, silent within-deadline hold,
back-pressure clock re-arm, backfill drain), and the gossip layer's loss boundary pinned on
a real two-endpoint iroh mesh (deaf-window recovery = exactly the rebroadcast ring; eviction
= permanent loss). Seconds-scale, no ceremony time; RQ-5's live-evidence residual unchanged.
**Fence:** no change specified here may land while a ceremony that pins the node binaries is in
flight (C2, run `f35bfa80…`, closed GREEN 2026-08-11 — no ceremony in flight as of Revision 6;
the fence re-arms with the next authored run). §7 (guest contract) additionally changes the
module hash **and the ABI minor** and therefore belongs to the next module revision (C3
lead-in), never to a frozen run's module.

**Motivating evidence.** Two C2 incidents in one night, both environment-caused, both
mis-handled by the current seams (C2 ledger, `~/experiments/ceremony-artifacts/c2-20260809/LEDGER.md`):
a failed `PayloadGet` completion trapped the tiny-llama guest
(`guest panic: tiny-llama/src/lib.rs:2814`), was classified a deterministic terminal, and —
compounded by the silent `WaitingForMembers` floor breach — idled the run for ~4.2 h until a
manual leave/rejoin; the same panic class recurred on a second box the same night. The
three-seat smoke recorded the identical pair as state-file non-claims 8a/8b. The precise fault
shape was established by Rung 0 (§2.1): the 23:09Z incident's failed completions were
**buffer-handle quota exhaustion (`GRANT_EXHAUSTED`), not a network fault** — the operator-era
"transient R2 egress fault" reading was wrong for that incident — while a genuine
transport-reset class fed the other guest-panic site the same night.

**Grounding discipline.** Every "today" claim below was verified against the working tree at
`851a2ef6`+docs (file:line cited inline). Where a draft rested on an unverified premise, the
correction is adopted (notably: retention is not replication; the node keeper already exists
and must not be duplicated; completion records ARE durably journaled — the tag-14 "reserved
(Phase B)" comment at `daemon-vhc-journal/src/record.rs:348` is stale documentation, not a
statement about behavior; and the working assumption that the C2 archive's completion codes
would be uniformly `STORE_REFUSED` was itself **falsified by the Rung 0 decode** — the seven
op arms all *map network faults* to `STORE_REFUSED`, but the archive's failed completions came
from three different producers with three different codes — §2.1).

**Companion documents.** [`vhc-architecture-spec.md`](vhc-architecture-spec.md) (tracked
architecture; §4.4 replay/verifiability is the invariant this spec must preserve),
[`vhc-module-abi-spec.md`](vhc-module-abi-spec.md) (normative ABI; §7 here is an ABI **minor
revision** landing there), [`vhc-fleet-ceremony-runbook.md`](vhc-fleet-ceremony-runbook.md)
(evidence gates; §6 adds a diagnosis entry), `../vhc-program-state.md` (status + non-claims;
8a/8b are annotated as each rung lands).

---

## 1. The conformance target

> **[REL-1] Target property: a guest trap implies a deterministic module defect.** This is NOT
> established by any single change in this document, and a host cannot make it globally true
> for arbitrary modules by changing supervision. It is a **conformance property** with four
> distinct obligations, each independently owned:
>
> 1. **Host obligation (§3, §4):** absorb transient environmental faults at the capability
>    boundary (bounded retry), and when absorption is exhausted, **name the permanent
>    truthfully** through the completion vocabulary.
> 2. **Guest obligation (§7):** handle every fallible environmental completion through
>    protocol vocabulary — a typed run-end outcome, never a trap. Panics stay reserved for
>    consensus/determinism invariants.
> 3. **Certification obligation (§7):** a module claiming conformance has had its
>    environment-sensitive panic paths audited out — a per-module property, checked at module
>    certification, not assumed.
> 4. **Compatibility bridge (§5):** for frozen, non-conformant modules, narrowly attribute
>    known environment-correlated traps as retryable — a bounded, auditable heuristic, not
>    proof.

The absorb / name-truthfully / react-in-protocol taxonomy is **permanent architecture**:
totality is unreachable (a committed payload can be genuinely gone — horizon expiry, credential
revocation, every source lost), so the truthful-failure path never becomes dead code. What is
NOT permanent doctrine is any inference from a bare trap to its cause (§5).

Liveness is the node's, not the guest's: consensus waits (e.g. the timeoutless
`WaitingForMembers`) stay timeoutless for safety, and the **node announces** stalls instead
(§6). No new trust authority; no node↔app WireVersion change; the only contract change is the
ABI minor in §7.

## 2. Seam map

```
R2/egress ──("REL-2: transient absorbed; expiry re-presigned")──▶ R2Store::get_content
    │ typed: Transient vs PayloadMiss vs expiry vs semantic refusal
    ▼
role_session op arms ──("REL-3: one shared mapper, honest COMP_ERR codes")──▶ completion (tag 14)
    │ delivered (tag 1)                                                           │
    ▼                                                                             ▼
frozen guest ──trap (tag 9)──▶ classify_trap ──("REL-4: bounded env heuristic")──▶ FailedRetryable
                                                                                      │
                                                  node keeper (EXISTING: reconverge, budget,
                                                  fresh incarnation, restore + catch-up)
                                                                                      │ exhaustion
                                                                                      ▼
VhcService tick ──("REL-5: Warning class=run_stalled")──▶ operator            FailedTerminal
```

Ordering constraint: **REL-3 lands before REL-4** — the attribution heuristic keys on honest
completion codes, which do not exist until the mapper does.

### 2.1 Rung 0 evidence record (C2 archive decoded, 2026-08-11)

The full C2 product archive (7,896 segments, 1,458,481 tag-14 completions across every role
chain) was scanned offline for every failure-shaped record: failed completions, trap/interrupt
terminals (tag-9 kind≠0), and conditions. The complete failure inventory:

| Failed-completion code | `detail` | Count | Fed which trap |
|---|---|---|---|
| 7 `GRANT_EXHAUSTED` | `buffer quota exhausted (deny new buffers)` (host buffer-handle mint refused, `pump.rs:932-935` et al.) | 74 | `lib.rs:2814` GuestPanic ×2 — trainer-2 inst 5 (the 23:09Z Windows incident, 3 refusals then trap) and trainer-1 inst 60 (~70 refusals over ~40 s of guest-side retrying, then trap) |
| 3 `STORE_REFUSED` | `det-state chunk <hash> fetch: transient transport fault (reset): read object body: error decoding response body` | 2 | `lib.rs:1834` GuestPanic ×2 (restore-window fetch; trainer-0 inst 226 and 238) |
| 4 `HASH_MISMATCH` | `sealed fold references an evicted chunk` (det-state store eviction, `state_store.rs:695-705`) | 2 | `lib.rs:1955` GuestPanic ×1 (round-base window fetch; trainer-1 inst 63) |

Trap terminals beyond the panics: `HostStorageExhausted` ×3 (`vhc disk quota exceeded …
4190370 B over the 64424509440 B quota` on trainer-0 inst 228/229 and trainer-1 inst 62;
`host free-space floor: … refused with 2419904512 B free (floor 2415919104 B)` on trainer-2
inst 7) and `GrantViolation` ×1 (`artifact ec95d3d6 is not in the admitted artifact set`,
trainer-2 inst 22) — §10's and §12's motivating incidents are now **journal-proven, not
operational readings**.

What this corrects, clause by clause:

- **The 23:09Z flagship incident is reattributed.** Its failed completions were host
  buffer-quota refusals (`st.buffers.create_host` returning `None`), not R2 egress. §3's GET
  retry would NOT have prevented it; a fresh incarnation (which resets the buffer pool) is
  what heals it, and the C3 guest obligation is backpressure handling: release handles /
  bounded wait on `GRANT_EXHAUSTED`, typed outcome on exhaustion — not a blind ~500 ms-cycle
  refetch loop, which is what the ~70-refusal trace shows the frozen module doing (§7).
- **The genuine transport class is pinned:** mid-body TCP reset on det-state chunk GETs — the
  same shape as the closure pulls' assembler failures (§3.1). No timeout, no 5xx, no
  throttling, and **no presign-expiry/403 evidence anywhere in the C2 record** — RQ-1's
  expiry lane is precautionary engineering against a real code defect, not incident-implicated
  (its empirical probe still gates REL-2's landing). The single-event reset shape also answers
  §3's pacing caveat for this population: no saturation signature, so bounded whole-object
  retry is the right first knob.
- **§5's whitelist would have fired on none of C2's panics as codes stand today** (observed
  codes: 7, 3, 4 — never 1/2). After §4's mapper lands, the reset class maps to the
  unreachable/timeout lane and becomes attributable; the `GRANT_EXHAUSTED` and evicted-chunk
  classes stay **deliberately outside** the heuristic (neither is network weather; both have
  their own rungs — §7 backpressure, RQ-11 retention) and their panics remain
  `FailedTerminal` until the C3 guest reacts in protocol vocabulary.
- **The round-46 gap non-heal is pinned to an unbounded respawn cycle** (RQ-5/RQ-6, §9): the
  C2 transcript carries 461 consecutive `co_trainer` warnings, every one
  `attempt 0 … paced respawn in 1000 ms`. Mechanism, verified: a successful co-trainer
  respawn removes the `co_retry` lane entry (`service.rs:1115-1116`), so the counter only
  counts *consecutive spawn refusals* — a flap cycle whose spawns succeed and whose sessions
  then die on the standing gap (≥ 20 s each, `GAP_DEADLINE`) runs at initial backoff forever:
  no growth, no budget, no escalation, no `min_uptime` discipline (the primary-seat keeper
  has all four). ≥ 2.7 h of continuous incarnation churn on the coordinator box. Endpoint/
  relay logs were not preserved past the ceremony, so the underlying iroh-vs-WS packet-loss
  mechanism remains unpinned — the RQ-5 residual.
- **A sixth failure class surfaced (RQ-11):** the det-state store evicted a chunk that a
  still-retained sealed fold references (`state_store.rs:688-705` — both the index-miss and
  spill-file-missing arms), surfacing to the guest as `HASH_MISMATCH` with an eviction detail.
  Two defects in one: the eviction/retention window contradicts the sealed-fold retention
  contract, and the code is dishonest (nothing mismatched — REL-3's mapper direction covers
  the coding half).

## 3. [REL-2] GET-side absorption at the store (the reliability foundation)

**Status: LANDED (2026-08-11), one recorded residual.** `get_content` is now
`get_content_once` (presign → GET → classify) wrapped in the PUT loop shape (`GET_ATTEMPTS =
4`, doubling backoff from 1 s — pacing from the §2.1 evidence: single mid-body resets, no
saturation signature). The normative matrix below is implemented exactly: 404 → prompt miss,
never retried; recognized-expiry 403 → `PresignExpired` on its own once-then-authoritative
lane; other 403 → semantic `Transport`, never a miss. The freshness guarantee is implemented
as invalidation: `PresignClient::presign_fresh` (default: forward; `HttpPresignClient`: drop
the cache entry, then mint), used by the expiry lane in BOTH the GET and PUT loops. Unit
coverage: transient-absorbed, miss-never-retries, expiry-refetches-on-`mint-1`-URL,
second-expiry-authoritative, plain-403-semantic, PUT-expiry-lane, cache-bypass — all
wiremock-fast, no live store. **Residual (RQ-1):** the expiry-body recognizer is the
conservative S3 vocabulary (`Request has expired`, `ExpiredToken`, `ExpiredRequest`); the
empirical probe of R2's actual aged-URL responses cannot be a short offline test (a URL must
age past its server-set TTL) and moves to the next ceremony's preflight, where an aged
presign is available for free. Until then an unrecognized R2 expiry shape degrades to the
semantic-403 lane — loud and typed, not looping, not a miss.

**Today (pre-landing state, kept for the record).** `put_content` retries transients — `PUT_ATTEMPTS = 4`, doubling backoff, fresh
presign per attempt, typed transient-vs-refusal discrimination
(`daemon-vhc-net/src/r2_store.rs:265-323`). `get_content` is one presign, one GET, no retry
(`r2_store.rs:294-308`). Below it, the object-GET path classifies
(`r2_store.rs:177-190`): 5xx/429/connect/timeout/reset → `VhcNetError::Transient`; **but 404
AND 403 both collapse to `Ok(None)` → `PayloadMiss`**. That 403 conflation contradicts the
crate's own taxonomy: `VhcNetError::PresignExpired` exists precisely because "the object may
well exist — the *credential* expired, so the caller must re-request a fresh presign rather
than treat the object as gone" (`daemon-vhc-net/src/lib.rs:214-219`), and the presign client
already re-mints once when the **presign endpoint** reports expiry (`presign.rs:320-326`).
The missing case is expiry reported by the **object endpoint**. Whether this fault class
played any part in the C2 incident is unknown until Rung 0 decodes the evidence; the
inconsistency stands on its own.

**Specified.** Extract `get_content_once` (presign → GET → classify) and wrap it in the PUT
loop shape: `GET_ATTEMPTS` bounded, doubling backoff, **fresh presign per attempt**.

Classification and retry matrix (normative):

| Object-endpoint result | Classify | Retry? |
|---|---|---|
| 5xx, 429, connect, timeout, mid-body reset | `Transient` | yes, bounded (the PUT precedent) |
| 404 | `PayloadMiss` | **never** — semantic availability; the stall ladder, the coordinator guest's evidence check, and reconstruct's budgeted refusal all rely on a **prompt** miss (`reconstruct.rs:320-329`) |
| 403 whose S3/R2 error body is a **recognized expiry** (e.g. `Request has expired`) | `PresignExpired` | re-presign and retry on its own bounded lane, NOT consuming the transient budget (mirrors `presign.rs:320-326`: once, then authoritative) |
| other 403 | semantic `Transport` refusal | never — an authorization failure is not "object absent" |
| other 4xx / malformed | `Transport` | never |

Do not classify all 403s alike: inspect the error response body. The exact recognized-expiry
matching (R2's actual error codes on an aged presigned GET) is established empirically before
landing — **RQ-1**. Note the 404→miss row also covers lifecycle expiry of the object itself;
that is correct and unchanged.

**"Fresh presign per attempt" must be made true, not assumed.** Re-calling `presign()` does
NOT guarantee a fresh URL today: `HttpPresignClient::cached()` returns the cached response
while it remains outside the local skew margin (`presign.rs:240-245`), so the object endpoint
can reject a credential the cache still considers live (clock skew; the store's own reading of
expiry) and a naive retry would re-present the identical rejected URL. Normative requirement:
the expiry-retry lane MUST be guaranteed a fresh credential — on a recognized expiry-shaped
403, invalidate that request's cache entry (or mint bypassing the cache) before the single
retry; a second expiry is authoritative and surfaces typed, mirroring `presign.rs:320-326`.
The mechanism (invalidation vs bypass-on-retry) is implementation choice; the freshness
guarantee is not. The same caveat applies to the existing PUT loop, whose "fresh presign per
attempt" comment (`r2_store.rs:271-272`) is weaker than stated for the identical reason —
repair it under the same contract. RQ-1 covers both the response-body recognition and the
cache behavior.

The retry lives **in `R2Store`**, not in a decorator and not in the session op handler: PUT
reliability already lives there (symmetry), and every GET caller — restore, reconstruct,
artifact fallback through the content plane — inherits it. A `PinnedArtifactStore`-style
payload decorator is rejected: its miss policy (retry-every-error, cache, second-store
fallback — `pinned_artifacts.rs:192-263`) is wrong for payloads, whose miss is caller policy.

### 3.1 [REL-2a] Offline assembler corollary: resume by re-verification

**Motivating evidence (C2 closure, 2026-08-11).** FOUR archive-pull attempts lost 30–50 minute
passes to single transients (one connect-timeout, three mid-body resets on distinct ~52 MB
payloads) against an egress path producing roughly one transient per 15–40 minutes — the REL-2
gap demonstrated in the offline tooling (C2 ledger, "Closure work").

**Status: LANDED (2026-08-11) — the durable version below, superseding the C2 field patch.**
Resume moved into the assembler itself: `fetch_verified_at` looks up every content object
local-first (destination file exists + re-hashes to the address ⇒ it IS the verified object),
counted separately in `AssembleReport` (`segments_reused`, `payloads_reused`,
`module_reused`) so a resumed assembly is visible in the report, not claimed as fetched. The
in-pass lineage double-fetch is absorbed by the same step (an in-pass cache hit, deliberately
NOT counted as reuse). All structural verification still runs every pass. The pull tool's
field-patch outer retry loop was REMOVED as redundant — the production
`R2Store::get_content` carries REL-2's bounded transient retry since Rung 1a, and stacking
the two would have made 20 silent attempts; the module-fallback bare GET gained its own
equivalent bounded loop (4 attempts, 1 s doubling, transient shapes only via the shared
egress classifier); and the fallback's error text now annotates rather than buries the
content-leg fault. Verified by the testkit assembly regression: a complete layout resumes
with the network unplugged (zero fetches, full reuse counts), and a tampered payload fails
the blake3 gate, is the ONLY object re-fetched, and is atomically repaired.

**Provenance (historical).** After the fourth failed closure pull the frozen-tool posture was
superseded mid-C2: `xtask/src/archive_pull.rs` (base `b43901ff6b3f`, diff
`sha256 c46c75a3…4ff9`, +39/−5 — full provenance in the C2 ledger) gained per-object bounded
retry on typed `Transient` faults and closure-level resume. Verification posture was
byte-identical. The certification verdict tool (`vhc-replay`) stayed the frozen commit; the
C2 VERDICT names the patched pull tool by commit + diff hash. That working-tree patch is now
superseded by the landed assembler-side version.

**Today (verified).** `assemble_archive` fetches unconditionally — the coordinator module
(`assemble.rs:200`), every sealed segment (`assemble.rs:215`), and every committed payload
(`assemble.rs:306`) — with no existence check against the output layout, and any single fetch
failure propagates through `fetch_verified`'s `?`, discarding the whole pass. Compounding it,
the coordinator-lineage segments are fetched **twice per pass**: once in the all-chains sweep
(`assemble.rs:215`) and again for record recovery (`assemble.rs:246`) — the lineage is a
subset of the same chains. The pull tool's fetch closure is the production
`R2Store::get_content` (`xtask/src/archive_pull.rs:63`), so REL-2's transient absorption is
inherited automatically when it lands — except the module-key **fallback** leg inside
`fetch_object`, which is a bare single `egress.get` (`archive_pull.rs:73`) and must be brought
under the same retry contract explicitly.

**Why resume is sound (and evidence-neutral).** The untrusted-store argument justifies
re-*verification*, never re-*download*: every artifact in the layout is content-addressed
(`segments/<hex>.seg`, `payloads/<hex>.bin`; `coordinator.wasm` is envelope-pinned), and
`write_atomic` is temp + fsync + rename (`assemble.rs:371-380`), so a final-named file cannot
be torn — an existing file that re-hashes to its address IS the verified object, judged by the
same `blake3` check a fresh fetch gets.

**Specified (the durable version — landed as summarized above).**

- Resume moves from the tool's closure into the assembler itself: `fetch_verified` gains the
  local-first step (if the destination file exists and re-hashes to the requested address,
  use it; on mismatch or absence, fetch as today), counted separately in `AssembleReport`
  (e.g. `payloads_reused`) so a resumed assembly is visible in the report, not silent — the
  field patch resumes but reports reused objects as fetched. Same verification, same refusal
  types, same layout. Assembler-side placement means every `assemble_archive` caller inherits
  it, not just this one xtask.
- All structural verification is unchanged and still runs every pass: head authorization,
  chain re-folds, `verify_head_binding` on every segment (existing or fetched), digest-conflict
  refusal. Resume never skips judgment, only bytes.
- The `archive_pull.rs` module-fallback GET joins the REL-2 retry contract (either routed
  through the store's retrying artifact path or given the same bounded loop).
- The `fetch_object` error text keeps the two legs distinct: today a transient on the content
  leg gets the module-fallback's (expected) 404 concatenated onto it, burying the real fault;
  report the content-leg error as primary and the fallback outcome as an annotation.
- Ranged single-object resume (continuing a mid-body reset from the received prefix) is NOT
  specified in this wave, but the closure pulls upgraded it from a footnote to an open
  question (**RQ-7**): four typed client-side failures (one connect-timeout, three mid-body
  resets on distinct ~52 MB payloads) from a **sequential, single-stream** fetcher. The
  single-stream fact partially answers §3's pacing caveat — no concurrency was present, so
  bounded whole-object retry IS the right first knob — and a ~52 MB re-fetch is cheap enough
  that ranged resume waits for evidence of retry thrashing on larger objects. A presigned-URL
  Range request also needs its own empirical verification before anyone assumes it works.

**Rung 0 (evidence before parameters) — EXECUTED, findings in §2.1.** The decode confirmed
the mechanics as qualified (details carry the typed `VhcNetError` display; the incident-window
transport faults were mid-body resets) and overturned the working assumption in one respect:
the panic-feeding completions were not all network-mapped `STORE_REFUSED` — the flagship
incident's were `GRANT_EXHAUSTED` buffer-quota refusals, outside this rung's seam entirely.
For THIS rung the operative findings are: the observed GET fault class is the transient
mid-body reset (bounded whole-object retry is the right first knob — no saturation signature,
single-stream contexts); and no presign-expiry shape appears anywhere in C2's record, so the
expiry-403 lane is precautionary correctness (the RQ-1 probe still gates landing) rather than
the incident's explanation.

## 4. [REL-3] One shared network-fault mapper (honest completion codes) — LANDED (2026-08-11)

**Before.** Seven session op arms collapsed every environmental error to
`COMP_ERR_STORE_REFUSED = 3`: `PayloadPut`, `PayloadGet`, `ArtifactFetch`, det-state chunk
reads, det-state covering span, artifact range open, artifact range fetch — erasing exactly
the distinction the store produces. The ABI already had vocabulary:
`COMP_ERR_NET_UNREACHABLE = 1`, `COMP_ERR_TIMEOUT = 2`, `COMP_ERR_STORE_REFUSED = 3`,
`COMP_ERR_HASH_MISMATCH = 4` (`daemon-vhc-abi/src/lib.rs:1458-1466`).

**Landed.** One shared mapper, `comp_error_code(&VhcNetError) -> u64`
(`role_session.rs`, beside `service_op`), reached by every environment-class arm through the
single failure constructor `net_op_failure(op_label, &err)` (op-labelled detail + host-side
`warn!` voicing) — not per-arm mapping, which is how the classification drifted to uniform
`STORE_REFUSED` in the first place. **RQ-2 resolved** — the exact assignment, validated
against ABI §7.5:

| `VhcNetError` | `COMP_ERR` | Rationale |
|---|---|---|
| `Transient { Timeout }` | `Timeout` (2) | the per-request deadline elapsed |
| `Transient { Connect / Reset / ServerFault / Other }` | `NetUnreachable` (1) | network / far end momentarily unavailable — environmental |
| `HashMismatch` | `HashMismatch` (4) | tamper/corruption reject path |
| `PayloadMiss` | `StoreRefused` (3) | authoritative absence / lifecycle — the §6.4 stall ladder's input |
| `Transport`, authoritative `PresignExpired` (post-REL-2 lane), `Fetch`, URL/scheme rejects | `StoreRefused` (3) | semantic refusals retrying cannot change |

The two HOST-side semantic refusals in the range arm (covering-span decompose failure,
span-fit refusal) deliberately stay explicit `StoreRefused` outside the mapper: no network was
involved. Exhausted-5xx lineage resolved to `NetUnreachable`, not `Timeout` — a 5xx/reset is
the far end refusing transiently, not a deadline elapsing. Unit tests pin the full assignment
exactly (`the_shared_net_mapper_assigns_the_abi_completion_codes_exactly`) so a drift
re-collapsing the vocabulary fails the suite.

Consequences (now in force): the journaled tag-14 completion carries the true fault class
(§5's attribution input and §8's audit evidence), and the C3 guest (§7) can react to the named
class. This changes completion **values**, which are journaled and replayed —
deterministic-safe (the journal records what the guest saw) and landed outside any ceremony
freeze. Note the C2-frozen tiny-llama predates honest codes: archives sealed before this
change still read uniform `STORE_REFUSED` (the §2.1 decode discipline stands for them).

## 5. [REL-4] Environmental trap attribution — a bounded compatibility heuristic — LANDED (2026-08-11)

**What this is.** Temporal correlation, not causal proof: the trap carries no op identity, so
"the guest trapped right after an environment-class failure" can never establish that the
failure caused the trap. It is specified anyway because it is the only recovery available to a
**frozen** module (the C2/C3-frozen tiny-llama cannot be patched), it is bounded by the retry
budget, and every application is durably recorded for audit. It is a supported compatibility
mechanism for non-conformant modules — subordinate to §7, which makes the guest react in
protocol vocabulary so no inference is needed. The recorded reason MUST label itself a
heuristic.

**Today.** `classify_trap` (`role_session.rs:2560-2581`) routes budget traps and
`ComputeFault` to `FailedRetryable`; **everything else including `GuestPanic` lands in the
catch-all `FailedTerminal { "module trapped: …" }`**. The keeper machinery downstream already
does everything the recovery needs and must not be duplicated (verified end-to-end):
`desired_state` stays `joined` on terminal (`handle_run_terminated` never changes intent,
`daemon-vhc-node/src/service.rs:2072-2309`); `runs_awaiting_retry` selects
`joined × failed_retryable` (`store.rs:753-759`); `reconverge` **mints a fresh incarnation**
(`service.rs:1234-1245`, REPLACE entry mode) and re-resolves restore + archive catch-up through
the unified late-join path (`resolve_restore`, `service.rs:3405-3530`) — the same recovery an
operator's `leave --immediate` + `join` performs; the budget bounds it (`max_retries 5`,
doubling backoff capped 60 s, `min_uptime_ms 30 s` reset, escalation to `FailedTerminal` on
exhaustion — `service.rs:976-1053`, `config.rs:202-225`).

**Implementation shape: extend the existing slice execution context, do not add a parallel
watermark.** The host already scopes traps to the exact delivered event slice, with the exact
stale-attribution discipline this heuristic needs: delivery activates the slice and stamps its
ordinal (`d.slice.slice_ordinal = Some(d.slice.slices_delivered)`,
`daemon-vhc-host/src/run/driver/linker/vhc.rs:225-228`); asking for the next event ends it —
"a trap in that window must not be attributed to the slice that already returned"
(`vhc.rs:57-60`); and trap handling already lifts a guest panic message **only on an exact
execution-context match** (`take_trap`, `lifecycle.rs:690-711`). Today this scaffold records
*which* slice trapped but not *what* that slice carried — the slice does not retain the
delivered completion's op/code/detail, and the `Trap` does not carry the ordinal. The
specified extension: when the delivered event is a completion, the activated slice retains its
metadata `(op, code, detail)`; trap classification reads it from the trap's captured context.
Adjacency and clearing then fall out of the mechanism itself — a trap between slices
(`slice_ordinal = None`) or in a later slice is unattributable by construction, rather than by
a reconstructed delivery-ordinal comparison.

**Landed exactly as specified above:** a failed completion's typed `(op, code, detail)` is
captured at enqueue (`EnvCompletion`, `pump.rs enqueue_completion` — from the typed
`CompletionResult`, never re-decoded from the frozen frame), moved onto the activated slice at
delivery and cleared at the same seam that ends the slice (`vhc.rs next_event`), attached to
the trap at the single trap-consumption point (`take_trap`), and consumed by the session's
`classify_trap` GuestPanic arm under the whitelist below. The trap journal record is
unchanged (the terminal record's fields are picked explicitly). Unit tests pin the whitelist
(both env classes attribute; `STORE_REFUSED`/`HASH_MISMATCH`/`GRANT_EXHAUSTED` and
evidence-free panics stay terminal) and the enqueue-side evidence capture.

A `GuestPanic` trap classifies `FailedRetryable` only when ALL hold (normative regardless of
mechanism):

- the trap's captured execution context is the slice that delivered the failed completion —
  no intervening delivery;
- the completion's code is environment-class per §4's mapper — `TIMEOUT` / `NET_UNREACHABLE`
  only, never `STORE_REFUSED` (a semantic refusal or genuine miss is not environmental
  weather);
- the recorded reason carries the evidence and the label:
  `"env-attributed trap (heuristic): op <N> failed <code> (<detail>); <trap>"`.

Everything else keeps today's arms. **Post-Rung-0 calibration (§2.1):** as codes stand today
this whitelist would have fired on none of C2's panics (observed panic-feeding codes were 7,
3, 4); after §4's mapper, the det-state reset class becomes `NET_UNREACHABLE`/`TIMEOUT` and
the `lib.rs:1834` incidents become attributable. The `GRANT_EXHAUSTED`-fed and
evicted-chunk-fed panics stay outside the whitelist by design — recycling does heal the
buffer-quota case (a fresh incarnation resets the pool), but attributing it here would paper
over a guest backpressure defect and a host retention defect that have their own rungs (§7,
RQ-11). The reason lands in `vhc_runs.terminal_reason`
(`store.rs:641-652`) and the archived journal independently carries the same record chain
(§8), so every attribution decision is **auditable, never silent**. Mis-attribution is
damage-bounded, not prevented: a genuinely divergent module re-traps, burns the budget, and
escalates to `FailedTerminal` — loud, not looping. Whether these constraints are the right
final rule (vs. trap-site metadata, guest-declared conformance, or accepting explicit
probabilism) is open — **RQ-3**; the constraints above are the floor, not the proof.

**The keeper is not touched.** Adding a second "retry failed-terminal" path is rejected: it
would erase the meaning of the terminal state and recreate the competing-recovery-paths
accretion that produced the c15 incarnation defects.

## 6. [REL-5] Node-level stall announcement

**Status: LANDED (2026-08-11).** The stall observer rides `reconcile_tick` exactly as
specified: per-run `ProgressTrack` carries **two watermarks** (committed progress from
`RoundOutcome`, local activity from `RoundProgress`/`CheckpointPublished`) resolving RQ-4's
structural half — the warning keys on committed progress, the detail reports both plus the
run head (`last_round`) and both ages. The watermark initializes at session readiness (the
run's own `RunPhase "running"`), so a run that never commits round 1 is detected.
`run_stalled` is voiced once per episode; the next committed round closes it with
`run_progress_resumed` — one stateful transition each way. **Deviation from the paragraph
below, recorded:** the per-run threshold is NOT derived from authored round policy (the
authored round wall is a guest/consensus-side value the node does not hold); it is
**adaptive** — `max(10 min floor, 2× the largest observed inter-commit gap for this run)` —
which self-calibrates to checkpoint-heavy rounds after one slow round has been observed.
RQ-4's residual (whether the floor and multiplier are the right defaults, and first-round
behavior before any gap has been observed) still wants a live checkpoint-heavy run before
freezing. Unit-test verified (episode open/close, single-voicing, adaptive threshold
stretch); no new event variant, warning class strings only, no wire change.

**Today.** A parked run is silent: consensus `WaitingForMembers` is not mirrored host-side
(`last_phase` stays `"running"` after attach; the phase enum lives guest-side,
`coordinator/state.rs:30-34`), a dead session emits nothing, and fleet liveness in C2 depended
on an ad-hoc log-scraping collector (whose block-buffering failure delayed incident response by
hours — evidence that observation must be product, not ops).

**Scope.** This detects the **still-running** side of an incident — exactly the C2 shape: the
trapped node went `FailedTerminal` (already a loud, typed, surfaced state needing no new
warning), while the coordinator and surviving trainer sat silently in `WaitingForMembers` for
hours. The warning belongs on those still-running nodes; it does not replace terminal-state
reporting and does not attempt recovery.

**Specified.** A stall observer on `VhcService`, piggybacking the existing `reconcile_tick`
(~5 s, `service.rs:955-969`): while intent is `joined`, the session is alive, and
age(progress watermark) exceeds a **per-run threshold derived from authored round policy**
(≈ 2× the authored round wall; never a single global constant), emit
`VhcEvent::Warning { class: "run_stalled", detail }` with the age, last round, and
peers/committed context — and a matching `"run_progress_resumed"` when progress returns: one
stateful transition each way, not a recurring alarm. The watermark **initializes at session
readiness** (so a run that never completes round 1 is detected) and updates on observed
progress. What counts as progress is a genuine design question — committed `RoundOutcome`
alone risks false positives across long checkpoint publications, while counting local
`RoundProgress` risks masking a real commit stall; the likely answer is multiple watermarks
(local activity vs. committed progress) with the warning keyed to committed progress and the
detail reporting both — **RQ-4**, to be validated against a live checkpoint-heavy run before
the threshold defaults freeze.

Normative constraints: a new Warning **class string**, never a new `VhcEvent` enum variant
(the six-arm union at `daemon-api/src/vhc.rs:348-418` is wire contract; a new arm is a CDDL +
codec + WireVersion change this rung does not need); not `Error` (a bounded observation, not a
failure); **no fake phase mirror** — the honest host-side predicate is
no-progress-while-joined-and-alive, not a guessed `WaitingForMembers`. Precedents:
`plane_health` (`role_session.rs:85-90`), `checkpoint_lag` (`service.rs:1972-2030`),
`seat_fenced` (`service.rs:856-894`). Operators and collectors consume it through the surfaces
they already poll (`vhc detail --watch/--json` → `recent_events`).

### 6.1 [REL-5a] Run-head progress vs session lifecycle — stop conflating them

**Status: LANDED (2026-08-11).** The session now writes honest lifecycle values into the
existing opaque phase string: `restoring` (a restore pointer is being resolved/rehydrated),
`catching_up` (a staged catch-up backlog is folding), `running`, `draining` (graceful leave
in progress). **Deviation, recorded:** the attached value stays `running`, not the
`attached` this section originally sketched — `"running"` is the exact string the node's
readiness promotion (Starting → Running) keys on, and it is now emitted only once restore
and catch-up have genuinely completed, which makes the promotion *more* honest rather than
renaming it. Display-only opaque string per the wire contract; no wire change.

**Motivating evidence (C2).** Both multi-hour delayed operator responses trace to telemetry
conflation: a restoring session reports `round=0` while the run head is at 40+;
`phase=running` covers restore, staged catch-up, and live attach indistinguishably; `peers`
semantics are ambiguous between roster size and gossip reachability. The operator cannot tell
"this box is healthy and catching up" from "this box is wedged" without reading logs.

**Today (verified).** The wire `phase` field is explicitly a display-only **opaque string**
("The node's last-known phase string for the run (display-only; opaque)",
`daemon-api/src/vhc.rs:222-223`) — richer values are NOT a wire change.

**Specified.** The host writes honest session-lifecycle values into the existing opaque
string — `restoring`, `catching_up`, `attached`, `draining` — from the states it already
transits (`resolve_restore` → staged catch-up → live attach), instead of one undifferentiated
`running`. Run-head progress (last committed round + its age) rides the REL-5 warning detail
and the existing event surface; a dedicated run-head field on the detail struct would be a
CDDL/WireVersion change and is NOT taken in this wave. `vhc detail` renders both axes
side by side from what it already receives.

## 7. [REL-6] Guest contract at the next module revision — an ABI minor — LANDED (2026-08-11, Rung 4)

*Landed exactly as specified below, all six bullets: ABI minor 6 + `OUTCOME_ENV_STARVED = 4`
(`daemon-vhc-abi`, spec §4.5 table + minor history in the same commit), the host
`FailedRetryable` arm plus the degrade-to-`Left` drift repair for reserved codes 5–15
(`role_session.rs`), the tiny-llama typed-run-end conversions with the classified panic-site
audit and bounded `GRANT_EXHAUSTED` backpressure (guest module hash changes — next-genesis
material, the frozen C2 artifact keeps non-claims 8a/8b), the 8b admitted-on-own-membership
fix, decay-while-waiting in the coordinator SDK tick (`Member.last_seen_s`, serde-defaulted;
staleness floored at phase start so restored rosters are never mass-dropped), and the runbook
§4.7 headroom generalization. Verification is unit-scoped by the post-C2 constraint: the
two-dead-trainer wedge and the window/liveness/reset scenarios are pinned at the
consensus-tick level (`decay_while_waiting.rs`), minor negotiation at the driver-selection
level, outcome arms at the session level. The whole-run testkit drills and the live G-3
shapes ride the next ceremony.*

This rung is an **ABI minor revision** (module-side only; no node↔app WireVersion change).
Outcome codes 4–15 are reserved "assigned only by a future minor of this document"
(ABI spec §4.5; `OUTCOME_STALE_RESTORE = 3`, `OUTCOME_MODULE_DEFINED_MIN = 16`,
`daemon-vhc-abi/src/lib.rs:1159-1161`), and the current minor is 5 (`DA_ABI_MINOR_V2 = 5`,
`lib.rs:94`). Assigning a code from the reserved range therefore requires, together:

- minor 5 → 6, with the ABI spec §4.5 outcome table amended in the same commit
  (`vhc-abi-spec-drift` gate);
- `OUTCOME_ENV_STARVED = 4`: "the run cannot proceed because a required committed input is
  unavailable after host absorption"; host maps it → `FailedRetryable` (a rejoin resolves a
  fresher restore pointer — exactly the `OUTCOME_STALE_RESTORE` shape, host arm
  `role_session.rs:2484-2489`);
- host support for minor 6 ships **before or with** any module declaring it;
- compatibility is already specified by the ABI (§4.5 OQ-5 note: a future-minor module never
  reaches an older host by §1.4 minor negotiation; an older host receiving code 4 anyway
  journals it and degrades to `Left`) — but that property is **verified in admission code and
  covered by a compatibility test**, not assumed from the spec text. **Known spec/code drift,
  recorded:** today's session maps every unmatched nonzero outcome to
  `FailedTerminal { "module ended with outcome {code}" }` (`role_session.rs:2491-2493`), NOT
  the ABI's degrade-to-`Left` reading. §1.4 negotiation shields the normal path, so this is
  conformance repair the compatibility test will expose (it only bites if negotiation is
  bypassed or buggy) — desirable alongside this rung, not a precondition for it.

Guest-side, in the same module revision:

- **tiny-llama fail-loud removal at the payload seam:** `completion_handle(&ev).expect("a
  record-listed committed payload fetches (fail loud)")` (`guests/tiny-llama/src/lib.rs:2813-2814`)
  becomes: a Failed completion on a record-listed payload ends the run with
  `OUTCOME_ENV_STARVED`. Rung 0 named this site's sibling seams concretely (§2.1): the
  restore-window fetch (`lib.rs:1834`) and round-base window fetch (`lib.rs:1955`) trapped on
  the same shape in C2 and convert with it.
- **Backpressure is protocol vocabulary, not a refetch loop (the actual 23:09Z class).**
  C2's flagship panic was fed by `GRANT_EXHAUSTED` buffer-quota refusals, and the frozen
  module's observed reaction was a ~500 ms-cycle blind refetch (~70 refusals in ~40 s) that
  cannot succeed while the guest itself holds the pool full — then a panic. The conformant
  guest treats `GRANT_EXHAUSTED` as backpressure: release consumed handles before re-request,
  bounded wait, and only after genuine exhaustion a typed run-end (`OUTCOME_ENV_STARVED`
  fits: a required input is unavailable to *this* incarnation). Host-side, nothing changes in
  this rung — the refusal is already honest and quota sizing is authored policy.
- **Panic-site audit, classified not counted:** the remaining `expect`/panic sites are
  **individually classified** — environment-sensitive seams (fallible completions:
  initialization fetches, restore, checkpoint publication) convert to typed outcomes;
  deterministic-invariant assertions (consensus safety checks on host-verified data)
  **remain panics** by design. A raw count of `expect`s is not the audit. Only after this
  audit does the module claim REL-1 conformance (obligation 3).
- **Non-claim 8b fix:** the unconditional `l.admitted = true` in the `RoundOpen` handler
  (`lib.rs:2478`) is **removed**; admission flips only on own-membership evidence — this
  peer's entry committed in an inline `RoundRecord` — and the 500 ms re-announce timer
  (`lib.rs:2453-2462`) keeps running until then, so an unadmitted rejoiner can heal a later
  roster vacancy. Future non-inline record-set encodings will need equivalent membership
  evidence extracted from the referenced set; that obligation travels with any such encoding
  change.
- **Membership decay must not freeze while waiting (the floor-breach wedge's root).** Zombie
  roster entries decay only inside round finalization — absence accounting lives in
  `finalize_round` ("account absences/drops", `coordinator/tick.rs:598-622`) — so once the
  floor breaches there are no rounds, hence no decay, hence a permanent wedge unless a fresh
  join fits the churn slot. C2 hit this repeatedly, including with TWO trainers dead at once
  (one churn slot is then insufficient). Guest-side fix in this module revision: membership
  maintenance ticks on the timer during `WaitingForMembers` — a peer whose staleness already
  exceeds the round-scaled absence equivalent decays without a round, freeing its seat for
  the announced rejoiner. Consensus safety is preserved because decay-while-waiting only
  *shrinks* the zombie set; it never admits anyone the Join path would not.
- **Runbook headroom rule generalized (authoring, not module):** `max_peers − min_peers ≥
  expected concurrent churn`, not a constant +1. C2's evidence: two dead trainers
  simultaneously, repeatedly. Runbook §4.7 is amended when this lands.
- Regression coverage: testkit whole-run for both behaviors, the minor-negotiation
  compatibility test above, decay-while-waiting + two-dead-trainer testkit scenarios, plus
  the existing G-3 drill shapes in the next ceremony.

## 8. Certification interaction (what the evidence chain does and does not prove)

The full causal-candidate chain of an environment-correlated trap is already durable in the
journal, in ordinal order (verified against `DurableSink`,
`daemon-vhc-session/src/journal_home.rs:561-598`):

| Order | Record | Carries |
|---|---|---|
| 1 | tag-14 `CompletionRec` (arrival) | op id + encoded `CompletionResult::Err { code, detail }` |
| 2 | tag-1 `EventRec` (delivery) | the identical completion frame the guest observed |
| 3 | tag-9 `TerminalRec` kind=1 (trap) | `TrapInfo { code, import, context, detail }` incl. the panic text |

This proves **sequence, not causation**: a completion was recorded, it was delivered, the
guest then trapped. What certification gains is therefore stated precisely: every §5
attribution decision is **independently auditable** — an offline verifier re-derives from the
archive alone that the heuristic's preconditions held (environment-class code, adjacency, no
intervening delivery) and checks the recorded reason against them. It does not re-derive "the
fault caused the trap". RED taxonomy still gains a lane: environment-correlated vs
module-diverged vs archive-corrupt. Caveats recorded: tag-14/1 ride non-committed appends and
become durable at the next commit barrier (the tag-9 terminal is `append_committed`); a
`MemorySink` worker (no journal home) leaves nothing on disk; a trap that precedes delivery
has tag-14 + tag-9 but no tag-1.

## 9. [REL-7] Records-transport seam: gaps are typed-retryable BY DESIGN; make them visible, rare, and cheap to survive

**Status: LANDED (2026-08-11) — clauses (a), (b), (d); (c)'s RQ-5 residual stays open by
design.** (a) The `plane_health` line gained a `gap[held= oldest_standing_ms= sender=]`
section computed by the 50 ms re-present pass: held-frame count, the age of the oldest gap
standing WITHOUT back-pressure (shadow passes report 0 — the defect-22 discipline carries
into the observability), and the gapped sender's peek — an impending `GAP_DEADLINE` verdict
is now observable live and in the end-of-session post-mortem line. (b) The ceremony runbook's
preflight (§4.2) gained the transport-posture item: multiple relay URLs per box (with the
single-`relay_url` roster constraint documented explicitly rather than extended in this
wave), `advertise_ips` direct addressing between mutually-reachable boxes, ledger entries for
in-flight config changes, and endpoint/relay log preservation (RQ-5's prerequisite). (d) The
co-trainer respawn lane is now CYCLE-bounded per the decided direction: a persistent per-run
ledger counts each sibling terminal against `max_retries` unless the session survived
`min_uptime_ms` (which resets it — the primary keeper's discipline); exhaustion parks the
lane with a loud warning naming the run-level lane as the escalation path, never a silent
infinite budget. `GAP_DEADLINE` is untouched. All three verified by unit tests (gap-snapshot
standing-vs-shadow, plane-health format, sustained-flap park + healthy-uptime reset).
RQ-6's open half (a distinct budget lane for transport-class vs guest faults) deliberately
stays open pending operational evidence.

**Motivating evidence (C2, 2026-08-11).** Windows trainer at round 46:
`state=failed_retryable retries=1 reason=inbound sequence gap unrecoverable (no backfill
within the deadline)`, concurrent with operator-observed iroh relay instability on Strix
(packet loss, dropped connections). The fault shape ("relay instability" vs endpoint vs
box-side network) is an operator observation, not yet pinned — evidence item below.

**Today (verified) — and unlike §3-§5's seams, classification here is already honest.** The
records channel is dual-plane — WS control plane + iroh gossip mesh, content-hash deduped
(`ControlPlaneStats`, `role_session.rs:96-107`) — so a standing gap means BOTH planes missed
the frames. Per-sender sequence gaps hold out-of-order frames (bounded at 4096,
`HELD_FRAMES_MAX`) and re-present them each 50 ms tick; a gap that stands **without
back-pressure** past `GAP_DEADLINE = 20 s` verdicts the typed retryable fault
(`role_session.rs:61-69, 1771-1811`; back-pressure shadows re-arm the clocks — the defect-22
lesson). That verdict flows to the existing keeper: fresh incarnation, checkpoint restore,
staged catch-up. The observed `retries=1` **is the designed recovery working** — the recovery
decomposition doc names in-session exact-frame repair as future work and warns "stretching
this deadline only postpones the same verdict; it is not the fix". On the iroh plane: relay
URLs are **node-local config**, not frozen genesis (`service.rs:2743`); each roster record
advertises only the FIRST relay URL (`service.rs:2766`); direct addresses come from the
`advertise_ips` node config; QUIC runs 120 s idle / 5 s keepalive with explicit
roster-seeded discovery.

**The reliability problem is therefore cost and frequency, not classification.** Every
standing 20 s gap burns a whole incarnation (restore + staged catch-up — minutes at ceremony
scale, and the round the fleet was in stalls meanwhile). And the keeper budget interacts
badly with sustained flap: `min_uptime_ms = 30 s` resets the retry count, so a
flap-die-restore cycle shorter than 30 s eats the 5-retry budget and lands `FailedTerminal` —
a silent roster hole until REL-5 (§6) ships the announcement.

Specified for this workstream:

- **(a) Gap-hold visibility on `plane_health`** (no wire change — richer detail on the
  existing warning class, 60 s cadence): held-frame count, age of the oldest **standing** gap
  (excluding back-pressure shadows), and the gapped sender. An impending 20 s verdict becomes
  observable, and post-mortems stop depending on node-log scraping. The end-of-session
  plane_health emission already provides the post-mortem hook.
- **(b) Relay/addressing posture hardening** (config + runbook, no code): pin **multiple**
  relay URLs in node config on every fleet box (the endpoint accepts a `RelayMap` set; if the
  roster record's single-`relay_url` field constrains peers to one dial-back relay, either
  extend the record or document the constraint explicitly), and provision `advertise_ips`
  direct addressing between fleet boxes wherever reachable, so the relay is NAT fallback
  rather than the hot path. Runbook preflight gains a transport-posture checklist item.
- **(c) Evidence (Rung 0 extension) — executed, partially resolved (§2.1).** What the record
  established: the non-heal was an **unbounded co-trainer respawn cycle** — 461 consecutive
  `attempt 0 … paced respawn in 1000 ms` warnings, ≥ 2.7 h of churn — because the `co_retry`
  lane's counter is cleared on every successful spawn (`service.rs:1115-1116`) and therefore
  only ever counts consecutive spawn *refusals*, never flap cycles; the sessions themselves
  died on the standing gap (≥ 20 s each). What the record could NOT establish (endpoint/relay
  logs were not preserved past the ceremony): the underlying iroh-vs-WS packet-loss mechanism
  behind the standing gap — that residual stays open as **RQ-5** and gains a prerequisite:
  (a)'s gap-hold visibility plus preserved endpoint logs on the next ceremony, so the next
  occurrence pins itself.
- **(d) Budget posture — the evidence upgraded this from "record a question" to "a defect
  with a decided direction".** The co-trainer lane today has NO cycle budget at all (§2.1) —
  the "silently infinite budget" this clause warned against is the shipped behavior on that
  lane. Direction: the co-trainer respawn lane adopts the primary keeper's discipline — a
  cycle budget with `min_uptime`-style reset (count a respawn against the budget unless the
  session survived its uptime threshold), exhaustion escalating loudly to the run-level lane.
  Whether transport-class faults then deserve a distinct (longer/slower) budget lane than
  guest faults remains **RQ-6**'s open half. Do not silently make any budget infinite; that
  hides a dead network behind an eternally-cycling seat.

**For the in-flight C2** (frozen binaries): the typed rejoin IS the designed recovery — let
it work. The relay list being node-local config means adding/changing relays or direct
addresses is an operational action that takes effect on the next incarnation without touching
the frozen run; any such change is a ledger entry.

**Classification verified; the observed non-healing has a different suspect.** The gap fault
provably classifies `FailedRetryable` (`transport_fault` → `classify_natural_end`,
`role_session.rs:2472-2474`) and enters the keeper. Where C2 nonetheless sat stalled for
hours after gap faults, the evidence points at the **storage seam blocking the keeper** (§10:
reconverge's re-assess refused on the disk floor and burned the budget), not at gap
misclassification — RQ-5's evidence pass must confirm this chain from the round-44/46
windows rather than assume either story.

**Desync regression lattice (2026-08-11, seconds-scale — no transport needed where the
transport adds nothing).** The missing-message behavior is pinned at its two honest seams:
(i) *the session's aging ladder* — `retry_held`'s core now runs over an abstract delivery
seam (`retry_held_with`, `role_session.rs`), because the session cannot tell iroh loss from
a relay flap from a WS drop: every desync converges to the same held-frame/gap-verdict
ladder. Unit tests pin the four faces in milliseconds: a standing gap past `GAP_DEADLINE`
returns the exact C2 verdict string; the same gap inside the deadline holds silently; a
back-pressure pass re-arms even already-expired clocks (the defect-22 shadow guard at the
ladder level); a late backfill drains the hold with no verdict however old it grew.
(ii) *the gossip layer's loss boundary* — `deaf_window_loss_is_bounded_by_the_rebroadcast_ring`
(`daemon-vhc-net/tests/iroh_gossip.rs`, real two-endpoint iroh mesh, ~1.3 s): a peer deaf
through N publishes recovers on rejoin EXACTLY the delivery-assurance ring's retained
messages via nonce-bumped re-floods, and everything evicted is permanently lost — iroh-gossip
has no anti-entropy, which is the mechanical reason the aged-gap verdict must be a rejoin,
not a wait. A relay-flap-specific drill was considered and deliberately not built: forcing
relay-only paths on loopback needs iroh's unstable custom path-selector internals, upstream
frames that facility as testing the relay protocol itself, and the deaf-window boundary
already covers the product-visible consequence regardless of what made the peer deaf. RQ-5's
live-evidence residual (preserved endpoint/relay logs on the next ceremony) is unchanged —
these tests pin behavior under loss, not which plane lost.

## 10. [REL-8] Storage lifecycle at the recovery seam

**Status: LANDED (2026-08-11, Rung 6).** All four clauses are implemented and unit-test
verified — no ceremony time consumed. (a) `reconverge_attempt` now runs the existing
run-scoped `reconcile_run_state_dirs` judgment before the fresh child's assess. (b) A
reconverge re-assess refusal carrying the free-disk lane-floor reason parks the run
`storage_gated` with a voiced `storage_gate` warning and an untouched retry budget; non-disk
assess refusals keep the budgeted lane (pinned by a node unit test against a floor-refusing
worker). (c) `xtask ceremony author` grew `--storage-budget-mb`/`--per-round-growth-mb` and
refuses an authored budget below `stop_rounds × growth + 25% restore headroom`; an unbounded
budget over a bounded reservation warns (pinned by an xtask unit test). Deviation: the
per-round growth figure is operator-supplied, not read from banked preflight evidence — that
automation is RQ-9's residual. (d) The reconcile tick reads the custodian's per-scope usage
and voices ONE `storage_pressure` warning per episode when a run scope crosses 80% of its
quota, clearing when usage recedes (pinned by a node unit test).

**Motivating evidence (C2 — as many manual interventions as the guest-panic class; now
journal-proven, §2.1: three `HostStorageExhausted` trap terminals carrying the quota figure
and one carrying the free-space floor refusal are in the product archive itself).** All
three boxes hit the ceremony's 60 GiB run scope through two distinct surfaces. (a)
*Recovery-blocking:* crash/rejoin cycles stacked each dead incarnation's restore
materialization — Windows accumulated ~59.8 GiB of dead recoverable state until reconverge's
re-assess refused `below lane floor: ram/disk` and the retry budget burned out against a full
disk; Strix later filled its scope the same way (~18 recycle incarnations overnight). (b)
*Healthy-session-killing:* M4 — the healthy trainer — died mid-run when its journal sink hit
the quota (`HostStorageExhausted`).

**Today (verified) — the machinery exists; the recovery path just never invokes it.**

- Reclamation is real and safety-disciplined: `daemon-vhc-node/src/reclaim.rs` deletes a
  **superseded** incarnation's spill state unconditionally and its journal only when the
  `CustodyLedger` proves every segment archived; the **newest** incarnation is never
  reclaimed (reconstruction input); `payload/` + `archive/` evidence planes are never touched
  (`reclaim.rs:62-172`). But it runs only at **startup** (`reconcile_orphans`,
  `service.rs:758-813`) and at **storage-gate open** (`service.rs:1143-1147`) — never on the
  reconverge path. `reconverge_attempt` → `resolve_restore` stages a fresh restore with zero
  reclamation first (`service.rs:1232-1386, 3404-3518`). The code itself documents the
  failure mode: "a quota-refused run once accumulated a dozen orphaned incarnations"
  (`service.rs:1136-1142`).
- The two refusals are different gates with different budget behavior — this asymmetry IS
  incident (a)'s mechanism. A journal-quota refusal (`FailedStorage`) parks the run
  **storage-gated** and defers WITHOUT burning retry budget until the gate opens and
  reclaims (`service.rs:993-999`). But `below lane floor: ram/disk` is the **admission
  funnel's free-disk probe** (probed free bytes vs the lane floor, default 10 GiB —
  `admission.rs:710-712, 123-131`; `backend.rs:2591-2600`) hit during reconverge's re-assess,
  and an assess refusal **does** burn the budget (`service.rs:1023-1037`) — so a disk filled
  by dead incarnations converts a recoverable environment condition into terminal escalation.
- The quota is operator/ceremony config, not genesis: `[vhc.storage] run_quota_mb`
  (default 0 = unbounded; `session/config.rs:237-266`), surfaced as the owner-budget disk
  charge (`service.rs:1734-1738`); the runbook's 60 GiB is seat-reservation sizing. **No
  authoring-time gate validates the storage budget against run length** — the only authored
  storage-adjacent gate is cadence↔retention (`det_state.rs:872-937`; `xtask/src/ceremony.rs:242-249`).
- Journal **Critical** writes (terminals, seals) are quota-exempt; **Normal** appends refuse
  at quota and surface `HostStorageExhausted` → `TerminalOutcome::FailedStorage`
  (`custody/lib.rs:336-362`; `journal_home.rs:303-315`; `role_session.rs:2549-2574`) — so
  incident (b) is a Normal-lane sizing failure, and the evidence-critical records still land.

**Specified.**

- **(a) Reclaim before every re-admission.** The reconverge path runs the existing run-scoped
  reclamation (the `reconcile_orphans` judgment: proven-superseded incarnations, spills
  unconditionally, ledger-proven journals) **before** the fresh child's assess — the same
  wiring `storage_gate_open_reclaiming` already has, moved onto the path that actually
  recycles incarnations. No new deletion logic, no change to the newest-kept and
  evidence-never-touched rules. This is the keeper-reuse argument of §5 applied to disk.
- **(b) Disk-floor refusals join the storage-gate lane, not the budget lane.** A reconverge
  re-assess refusal whose reason is the free-disk lane floor parks the run `storage_gated`
  (reclaim → re-check → resume), exactly like `FailedStorage` does today, instead of burning
  retry budget. A floor breach that survives reclamation (the disk is genuinely full of
  evidence + the newest incarnation) still escalates loudly — the gate re-check refuses and
  says why. Non-disk assess refusals keep today's budgeted lane.
- **(c) The authoring gate the quota needs (the §4.7 precedent).** Ceremony authoring
  validates `storage budget ≥ stop_rounds × observed per-round growth + restore headroom`,
  with the per-round journal+payload growth figure taken from banked preflight evidence (the
  fit-probe / smoke transcript) the runbook already requires — where the figure comes from
  and its safety margin is **RQ-9**. An unbounded quota (0) on a bounded seat reservation is
  an authoring warning: the effective bound is then the disk, discovered at the lane floor.
- **(d) Pressure is announced before it kills.** The custodian already computes
  pressure/free-floor state; surface it as a `storage_pressure` warning class (REL-5's
  precedent stack) when a run scope crosses a threshold of its quota, so (b)-class deaths
  stop being surprises. No wire change (warning class string).

## 11. [REL-9] Keeper reaction: recycle stalled-but-alive sessions, stand down completed runs

**Status: LANDED (2026-08-11, Rung 7).** Both clauses are implemented and unit-test verified
— no ceremony time consumed. (a) A stall announced by REL-5 that persists past a separate,
larger reaction threshold (3× the announce threshold) becomes a PACED external-progress
probe (one archive-head verification per minute at most, never a per-tick registry poll);
when the run's VERIFIED head round claim has advanced past this session's last committed
round, the keeper voices `stall_recycle` with the evidence in the reason and ends the
session through the ordinary retryable terminal lane — existing budget, backoff, escalation,
instance released for reconverge (pinned by trigger-arithmetic and lane unit tests). Only a
session that HAS committed rounds is eligible — the RQ-8 whole-run-wedge exception is
deliberately not taken. (b) Before any retry is spent, the reconcile pass checks
run-terminal evidence — the registry descriptor's authored total-round count (the new
`RunDiscovery::run_rounds` seam) against the verified archive-head round claim
(`rounds_done ≥ stop`); a provably-over run stands down to the deliberate-end lane
(`Completed`, retry cleared, `completion_stand_down` voiced with the evidence) instead of
cycling (pinned by an integration test over a real frozen genesis + signed head, including
the mid-run and no-stop-figure negatives). Descriptor metadata alone never proves progress —
the round claim is signed, certificate-chained evidence; absence of evidence is never
"over".

**Motivating evidence (C2).** Roughly eight manual `leave --immediate` + `join` interventions,
each performing exactly what the keeper's reconverge already does, triggered by a human
watching a stalled watcher line. And after the run completed, Windows' last incarnation kept
`failed_retryable`-cycling against a finished run — verified: nothing on the retry path
checks run-terminal evidence (`runs_awaiting_retry` selects `joined × failed_retryable` with
a time gate only, `store.rs:753-759`; `reconverge_attempt` consults nothing about the run's
completion, `service.rs:1232-1290`).

**Position.** REL-5 announces; this rung acts — and §5's warning against competing-recovery
accretion stands, so the ONLY action this rung may take is the one the keeper already owns:
end the session typed-retryable so the existing reconverge lane (budget, backoff,
`min_uptime` reset, escalation) recycles the incarnation. No second recovery path, no new
budget, no consensus-side action.

**Specified.**

- **(a) Bounded stall-recycle.** When the REL-5 stalled state has persisted past a separate,
  larger reaction threshold AND the node has evidence the run is progressing without this
  session — the run's archive heads advancing past the session's last committed round (the
  node already reads archive heads for catch-up) — the keeper ends the session with a typed
  retryable reason (`stall_recycle: session made no progress while the run head advanced`)
  and lets reconverge do what the operator's leave+join did. The external-progress condition
  is the guard against fleet-wide self-recycling when the run is legitimately waiting (a
  floor breach stalls EVERYONE's head — recycling the healthy coordinator helps nothing);
  whether a bounded exception exists for the 8b-silence shape (whole run wedged, zombie seat
  holder is *this* box) is **RQ-8**, decided with C3's decay-while-waiting fix (§7) in view —
  that fix removes most of the 8b-wedge class at the source.
- **(b) Completion stand-down.** Before scheduling any retry, the keeper checks run-terminal
  evidence (registry descriptor status / a terminal archive head); if the run is over, the
  intent stands down to a deliberate end (no respawn — the same "deliberate end" lane that
  already never respawns, `service.rs:5098`). A completed run is not an error state to retry
  against.
- Both actions are journaled with evidence-carrying reasons (the §5/§8 audit discipline) and
  bounded by the existing retry budget — exhaustion still escalates loudly.

## 12. [REL-10] Grant staleness under checkpoint rotation

**Status: LANDED (2026-08-11, Rung 8).** The continuous case is closed at the one seam where
committed run evidence is minted deterministically: a `net@2::payload_put` whose bytes carry
the host's §10.2 checkpoint-document shape has its ByRef family folds inserted into the
putting incarnation's `granted_artifacts` (`linker/net.rs`, `checkpoint_evidence_folds`) —
the put CALL is guest output and journaled, so replay reproduces the identical extension at
the identical point. Any other payload (including CBOR that is not the doc shape) mints
nothing; a fetch of a hash with NO committed evidence still traps `GrantViolation`
(deterministic defect lane, unchanged). The normative `grants.cddl` drift is repaired: the
optional `artifacts` field the Rust `GrantsDoc` already carried is now in the schema,
annotated that runtime evidence extension is host memory and never re-encoded into the
document. Verified by unit tests (fold extraction: doc shape with mixed inline/by-ref
sections, arbitrary bytes, non-doc CBOR; grammar: `grants-doc` with `artifacts` validates).

**Motivating evidence (C2, a wholly new defect class — journal-proven, §2.1: the tag-9
`GrantViolation` trap for artifact `ec95d3d6` is in the product archive, trainer-2 inst 22).**
The guest trapped
`GrantViolation: artifact … is not in the admitted artifact set` when checkpoint rotation
minted an artifact after its admission-time grant was frozen. Checkpoint publication is
authored, periodic behavior — every long-lived incarnation is a future violation waiting for
a cadence boundary.

**Today (verified).** The admitted artifact set is a **hash-enumerated allow-list frozen at
admission**: genesis role grants (`genesis.rs:279-281, 490-494`) → admission replaces
`RunConfig.granted_artifacts` (`admission.rs:544-563`) → cloned into the instance at start
(`lifecycle.rs:340`). It gates `data@2::fetch` / `register_chunks` / `register_state_chunks`
(`linker/data.rs:44-72, 241-300`) — not `PayloadGet` (different plane). Two relief valves
exist and define the intended trust shape: a **self-sealed** fold bypasses the grant while
retained ([SF-R1], `data.rs:47-57`), and `da_migrate` inserts the verified checkpoint
capture's ByRef family folds — post-admission hashes granted from **committed run evidence**
(`lifecycle.rs:539-545`). Nothing extends the set for a *continuing* incarnation as rotation
mints fresh folds; upgrade re-admission may only shrink it (`role_session.rs:1483-1490`).
**Known spec/code drift, recorded:** the normative `grants.cddl` omits the `artifacts` field
the Rust `GrantsDoc` carries (`grants.rs:62-63`) — the implementation is ahead of the CDDL;
repair the schema alongside this rung.

**Specified.** Generalize the `da_migrate` precedent into the rule it already embodies:

> An artifact hash named by **verified, committed run evidence** (a ByRef family fold in a
> committed checkpoint document of this run) is granted to the incarnation that observes that
> commitment — at migrate time (today's one-shot) AND on subsequent committed checkpoint-doc
> observations (the missing continuous case).

This is host-side state only (the set is host memory, not wire or ABI), preserves the
grant's security intent — the module still touches only content the run itself committed,
extended by exactly the evidence class `da_migrate` already trusts — and never widens genesis
grants for foreign artifacts. The trap itself stays: a fetch of a hash with NO committed
evidence remains a genuine `GrantViolation` (deterministic defect lane). The longer-term
vocabulary question — first-class grant *classes/planes* (e.g. "this run's checkpoint plane")
in the ABI grants document instead of hash enumeration — is an ABI-minor design item deferred
to C3 scoping, **RQ-10**.

## 13. Named future capabilities (recorded, not built)

**Exact-frame archive backfill (the transport seam's principled lane — recovery invariant
(1)).** The fault string's "no backfill within the deadline" names a lane that does not exist
yet: the recovery decomposition (`role_session.rs:61-69`) separates (1) transport sequence
repair, (2) semantic record catch-up (exists, narrowly), (3) module/journal reconstruction —
and every role session with a durable journal home already publishes its frames per seal
(ABI §8.8 incremental publisher). A receiver with a standing gap could therefore fetch the
gapped sender's sealed segments from the archive and replay the exact missing
`(sender, channel, seq)` frames in place, ending the gap without burning the incarnation —
bounded by seal cadence (the unsealed tail cannot backfill; the typed-retryable rejoin
remains the floor beneath it). This composes with REL-2: the backfill fetch rides the same
retrying content plane. Design note only in this wave; it is the durable answer to
"rejoin-per-gap is too expensive", where deadline stretching is explicitly not.

**Multi-source content plane.** "Failover to another copy" is honest future work, not a
composition of existing pieces: under the current topology R2 holds the only copy (retention
is a genesis-time cadence-versus-horizon gate, `det_state.rs:879-933` — not replication), and
the production session refuses every direct-stream op (`role_session.rs:2220-2226`). What
exists: the pump's complete stream plumbing (credit accounting, verbatim journaling,
`pump.rs:1136-1169`) and a written-but-dead multi-source fetch policy bound to the harness-era
`PayloadStore` seam with **inverted** retry semantics (`fetch.rs`; callers: tests only).
Deciding whether `fetch.rs` is rebound to `ContentStore` or deleted is part of this
workstream's scope when it opens; leaving a written-but-unwired policy on a superseded seam is
its own small debt. Also in scope then: peer-served payloads over iroh (transport binding +
serving/authorization policy are the genuinely new parts) and a payload cache as a
cost-ordered source.

## 14. Where implementation edits land (documentation taxonomy)

| Change | Lands in (same commit as the code) |
|---|---|
| REL-2/-3 seam behavior | this spec flips clauses from "specified" to "landed" (RQ-1/RQ-2 resolved inline); runbook §5 preflight/diagnosis notes if operator-visible |
| REL-2a assembler resume | this spec; runbook §3.4 archive-pull notes (resumable assembly, `*_reused` report fields) |
| REL-4 heuristic | this spec; `vhc-program-state.md` non-claim 8a annotated with the mitigation and its heuristic character |
| REL-5 warning class | runbook §5 (diagnosis entry beside plane_health); this spec (RQ-4 resolved inline) |
| REL-7 gap visibility + transport posture | this spec (RQ-5/RQ-6 resolved inline); runbook preflight transport-posture item + §5 diagnosis notes; ledger entries for any in-flight config changes |
| REL-8 storage lifecycle | this spec (RQ-9 resolved inline); runbook §4.7-adjacent authoring gate + storage-posture preflight; `vhc-program-state.md` storage non-claim |
| REL-9 keeper reaction | this spec (RQ-8 resolved inline); runbook §5 diagnosis (recycle + stand-down reasons) |
| REL-10 grant extension | this spec (RQ-10 deferred to C3 scoping); `grants.cddl` artifacts-field drift repaired beside it; ABI spec §2.6/§12.14 notes if wording shifts |
| REL-5a phase values | this spec; runbook §5 (what each phase value means during diagnosis) |
| REL-6 membership decay + headroom | `vhc-module-abi-spec.md` if the waiting-decay rule is normativized; runbook §4.7 headroom rule amendment |
| REL-6 ABI minor 6 | `vhc-module-abi-spec.md` §4.5 outcome table + minor history (normative, same commit as the constant — `vhc-abi-spec-drift` gate); non-claims 8a/8b closed or narrowed in `vhc-program-state.md`; architecture spec §4.4 note on the trap-taxonomy lane |
| Status at every boundary | `vhc-program-state.md` §7/§8 + the program handoff document |

## 15. Non-goals

No new wire certificate or signing authority; no consensus-side timeouts (safety stays
guest-side); no keeper duplication or retry-of-terminal path (REL-9 acts only through the
existing reconverge lane); no payload-plane cache decorator in this wave; no R2 retention
**enforcement** (remote retention remains a standing non-claim); no standby coordinator
failover; no module changes to the frozen C2 tiny-llama; no claim that REL-1 holds for
arbitrary third-party modules — it is claimed per module, at certification, after the §7
audit; no exact-frame archive backfill built in this wave (§13 records it); no `GAP_DEADLINE`
stretching (the code's own doctrine: it postpones the same verdict); no silent unbounding of
the keeper retry budget; no ranged partial-object resume in the assembler (§3.1 records when
to revisit); no reclamation of the newest incarnation or the `payload/`/`archive/` evidence
planes — REL-8 changes WHEN the existing judgment runs, never what it may delete; no grant
widening beyond committed run evidence (REL-10 extends by exactly the class `da_migrate`
already trusts). The original "no tool changes during C2 closure" fence was **superseded in
the field** for the pull tool after four failed frozen-tool attempts (§3.1) — the exception
is bounded (offline maintainer tool only, verification-neutral, provenance recorded in the
C2 ledger by commit + diff hash) and the fence stands unchanged for node binaries, frozen
artifacts, and the verdict-producing `vhc-replay`.

## 16. Open design questions (RQ register)

| RQ | Question | Resolved by |
|---|---|---|
| RQ-1 | **Narrowed (§3 landed):** the freshness mechanism is decided and landed (cache invalidation via `presign_fresh`); the conservative S3-vocabulary recognizer is landed. REMAINING: which error bodies R2 actually returns for an aged presigned GET (and whether expiry is enforced at admission or mid-transfer) — an unrecognized shape currently degrades to the loud semantic-403 lane. *Rung 0 (§2.1): no expiry shape anywhere in C2's record — precautionary, not the incident class* | aged-URL probe at the next ceremony's preflight (a URL must age past its server-set TTL — not a short offline test) |
| RQ-2 | **Resolved (§4 landed):** the exact assignment is pinned in §4's table and by unit test — `Transient{Timeout}`→`Timeout`; every other transient (connect/reset/5xx/other)→`NetUnreachable` (a 5xx/reset is the far end refusing transiently, not a deadline elapsing); `HashMismatch`→`HashMismatch`; miss + every semantic refusal (incl. authoritative post-REL-2 `PresignExpired`)→`StoreRefused` | landed with REL-3 (validated against ABI §7.5) |
| RQ-3 | Whether §5's minimum constraints are the right final attribution rule, or trap-site metadata / guest-declared conformance / explicit probabilism is needed | operational evidence from the first runs with REL-4 active; revisited at C3 certification |
| RQ-4 | **Structurally resolved (§6 LANDED):** dual watermarks, warning keyed to committed progress, detail reports both; threshold adapts to the observed inter-commit gap (2×) over a 10-min floor. RESIDUAL: are the floor/multiplier the right defaults, and is first-round behavior (no observed gap yet) acceptable | validation against a live checkpoint-heavy run before defaults freeze |
| RQ-5 | **Partially resolved (§2.1/§9c):** the non-heal mechanism was the unbounded co-trainer respawn cycle, pinned. RESIDUAL: the underlying iroh-vs-WS packet loss behind the standing gap (endpoint logs were not preserved) | §9(a) gap-hold visibility + preserved endpoint logs on the next ceremony |
| RQ-6 | **Decided half IMPLEMENTED (§9d LANDED):** the co-trainer cycle budget ships — `max_retries` cycles, `min_uptime`-reset, loud park on exhaustion. OPEN half: whether transport-class faults then deserve a distinct (longer/slower) lane than guest faults | operational evidence with REL-5 announcing exhaustion |
| RQ-7 | Ranged single-object GET resume for large payloads (mid-body resets on ~52 MB objects observed; presigned-URL Range semantics unverified; whole-object retry currently cheap enough) | revisit on evidence of retry thrashing; empirical Range probe first |
| RQ-8 | **Decided conservatively (§11 LANDED):** external run-head progress is the ONLY reaction condition taken — a never-committed session is never recycled, and the whole-run-wedged 8b shape (zombie seat holder is this box) is deliberately left to the operator today. REMAINING: whether a bounded exception is worth taking at all — narrower now that §7's decay-while-waiting is LANDED (Revision 15), which removes most of that wedge class at the source (in the NEXT module revision; the frozen C2 module keeps the wedge) | operational evidence from the first ceremony running the minor-6 module |
| RQ-9 | **Structurally resolved (§10 LANDED):** the gate ships — `budget ≥ stop_rounds × growth + 25% restore headroom`, unbounded-budget warning. RESIDUAL: the growth figure is operator-supplied; sourcing it from banked preflight evidence (fit-probe vs smoke transcript) and freezing the 25% headroom margin remain open | banked preflight evidence from C2 + the three-seat smoke |
| RQ-10 | First-class grant classes/planes in the ABI grants document (vs hash enumeration + evidence-based extension) | C3 ABI-minor scoping; REL-10's host-side extension (§12 LANDED) is sufficient until then |
| RQ-11 | The det-state eviction/retention contradiction (§2.1): the store evicted a chunk a still-retained sealed fold references (`state_store.rs:688-705`), and the guest saw `HASH_MISMATCH` for a fault that mismatched nothing. Two halves: the retention-window defect (why did eviction run ahead of the sealed-fold reference?) and the dishonest code (REL-3's mapper direction) | retention-window audit of `state_store` eviction against the sealed-fold retention contract; code honesty lands with REL-3 |
