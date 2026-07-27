# daemon-vhc — normative module ABI specification (v2)

**Status: NORMATIVE, and this tracked copy is the only normative one.** Imported into the repository
2026-07-26 as the canonical module ABI, per the ratified placement of normative amendment A5. It is the
companion [`vhc-architecture-spec.md`](vhc-architecture-spec.md) has always named in its header, and
it now exists where the code that implements it lives: it versions with that code and travels inside
every certification candidate.

**The external copy is demoted in the same change.** `daemon-vhc-abi-spec.md` in the program directory
is a **non-normative pointer** naming this path as authoritative. Two hand-maintained normative copies
would violate [DI-3] — in the same ratification that introduces [DI-3] — for the most consequential
shared document in the program, so there is exactly one, and it is this file.

**Drift is checked mechanically, against the code rather than against a copy.** A text mirror can only
drift from its source; a specification can drift from the *implementation*, which is the failure that
costs something. `xtask vhc-abi-spec-drift` (in the mandatory lane) asserts that the version ladder,
export names, journal tags and bound constants stated below are the values the `daemon-vhc-abi` crate
actually defines. A statement in this document that the code contradicts fails the gate.

This document fixes *the wire*: version negotiation, the module entry contract, event-frame encoding,
the journal record format, the resource-plan and claim schemas, resource/handle semantics, the trap
taxonomy, the migration scaffolding, and the guest↔host threading contract.

Where this document and [`vhc-architecture-spec.md`](vhc-architecture-spec.md) disagree, that is a
**defect in this document**; the architecture governs.

---

> ## Reading order for the certification minor
>
> The body of this document fixes major 2 **minor 0** and the additive minors above it. **§17 carries
> the certification minor (major 2, minor 5)** and is normative where it and the body differ: it
> replaces §9's `claim()` surface, extends §6.5's `log` bounds, and replaces §7.6's context
> enumeration with a minor-selected one. Every other subject in the body — version negotiation, the
> module entry contract, event framing, the journal record format, resource and handle semantics, the
> trap taxonomy, migration scaffolding, the threading contract — is unchanged and remains the
> reference.
>
> The sections §17 supersedes carry a pointer at their head, so a reader who arrives at one of them
> directly is not left with the superseded text.

---

## 0. Conformance, terminology, and structure

### 0.1 Requirement levels

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHOULD**, **SHOULD NOT**, **MAY**,
and **OPTIONAL** are to be interpreted as in RFC 2119/8174. A conforming **host** and a conforming
**module** each satisfy the clauses addressed to it. A clause addressed to "the ABI" constrains both
sides and their shared contract crate (`daemon-vhc-abi`).

### 0.2 Roles of the actors

- **Module / guest** — a `wasm32-unknown-unknown` blob, pinned by blake3, running one role-instance
  (architecture §10 "one role-instance per sandbox"). Untrusted.
- **Host** — the `vhc-worker` process: sandbox, capability providers, journal, signing oracle.
  Trusted locally, adversarial to the network.
- **Session** — the host component that owns the event pump, delivery classes, and the journal
  (`host/daemon-vhc-session`).
- **Contract crate** — `daemon-vhc-abi`: the single Rust definition of world/import names+versions,
  handle types, the event set, the trap vocabulary, the channel table, and the
  manifest/claim/journal schemas, asserted identical on both sides (architecture §7, dependency
  rule 3; refactor §3 permitted extraction).

### 0.3 Canonical encoding

Every structured value that crosses the boundary as bytes — the manifest, the claim, event frames,
config, grants, the journal — is **canonical CBOR** as already defined by
`daemon-vhc-proto::canonical` (deterministic map key ordering, shortest-form integers, no
indefinite-length items, no floating NaN/Inf in canonical positions). "Canonical CBOR" below always
means that encoding. Two encodings of the same value MUST be byte-identical; this is the property
replay, hashing, and signatures depend on.

Wire-schema conventions used throughout: **hot per-event structures are definite-length positional
arrays** whose first element is an assigned numeric tag; **descriptive structures (manifests,
declarations, journal record bodies) are maps with explicit string keys**. All enumerations carry
assigned numeric values fixed in this document; no enumeration is "to be pinned later" anywhere
journal bytes are affected.

### 0.4 Versioning of this document

Sections are independently versioned by the ABI major/minor they describe. The v2 surface defined
here is **`da_abi` major 2, minor 0**. Section numbering is stable; additive amendments raise the
minor and append (never renumber). The `compute@` tensor surface (architecture §3.2, "an explicit
architecture gate") is **out of scope for Phase A** except for the transitional `tabi@1` bridge
(§2.5); the v2-native compute world is ratified separately as the Phase C entry gate.

### 0.5 Changelog

- **Imported as the canonical tracked ABI, and minor 5 added (2026-07-27).** This document moved into
  the repository as the single normative module ABI, per the ratified placement of normative amendment
  A5; the external copy in the program directory became a non-normative pointer in the same change.
  **§17** is new and carries the certification minor (major 2, minor 5): `da_resource_plan` and
  `da_apply_execution_grant` with their exact ownership and typed refusals, the derived derivation
  budget, the minor-selected tag-0 schema and tag 18, the eleven-value typed execution context with
  minor-selected rendering, the bounded `sys@2::log` exemption during `da_init`/`da_migrate`, the
  profile-keyed lane claim bounds, and the rule that a permitted revision range names whose numbering
  it constrains. §6.5, §7.6 and §9 gained pointers to what §17 supersedes; nothing else in the body was
  rewritten. Decision codes of the retired program vocabulary were scrubbed; `[SF-n]` **rule**
  identifiers are legitimate specification ids that tracked Rust cites and are preserved verbatim.
  Drift against the implementation is gated by `xtask vhc-abi-spec-drift`.

- **Draft 3 — amendment a13 (2026-07-23, the checkpoint/restore wave — by-reference document v2 +
  streaming rehydration; label unchanged).** §12.14 [SF-6] promotes from "shape fixed" to
  **normative as implemented**, shapes unchanged: the checkpoint document carries
  `master`/`ef`/`adamw_m`/`adamw_v` as `FamilyRef`s + the inline `round`, under the shared
  `[manifest_bytes, [ckpt-doc-section…]]` codec; a live checkpoint moves zero extra local bytes
  (the families are already sealed). Restore is streaming rehydration: the §10.2 migration
  descriptor's `migration-section` gains the untagged by-reference alternative (`{name, family}`)
  beside `{name, staging_id}`; `da_migrate` records the refs (no bulk read — §6.6 unchanged) and
  `da_run` registers each fold ([SF-R2]) and streams its windows with bounded in-flight refill.
  The DRAIN carries by-ref sections reconstructed host-side from the draining instance's state
  store. Cadence/publisher policy: local cadence gates the boundary, remote cadence + a
  one-per-slot deterministic publisher election gate the upload; the publisher uploads the
  document + its family chunks content-addressed (idempotent — unchanged chunks upload nothing);
  the cadence↔retention bound is enforced at genesis authoring; referenced folds are
  retention-pinned while freshest of their (role, kind) slot. **No op shape, trap code, journal
  kind, grant bound, or wire-version change**: the descriptor + document are `VhcProtoVersion`-2
  shapes (already bumped by [SF-5], a12); the `MigrationSection` by-ref alternative and the host
  `OpRequest` chunk-keyed det-state range resolution are internal seams, not ABI/wire surfaces;
  app↔node `WireVersion` unchanged. Existing pinned digests are reproduced bit-for-bit
  (checkpoint-migrate golden re-pinned to the by-ref FORM, digest VALUES unchanged).
- **Draft 3 — amendment a12 (2026-07-23, the streaming trainer guest + externally-sourced-root
  reads; label unchanged).** The flagship trainer guest is rewritten onto the streaming engine
  with NO resident canonical state: the guest holds no master/round-base/error-feedback vectors —
  ingest, make_update, quiesce and the live checkpoint all read/write the host-side sealed folds
  through the state ops, and the streamed path is the ONLY path (the 64-dim ceremony run
  reproduces the pinned digests bit-for-bit under the production `EngineConfig`, with the recorded digest evidence).
  §12.14 [SF-3]'s [SF-R2] and [SF-R3] promote from "reserved" to **normative as implemented**:
  [SF-R2] externally-sourced roots register through a new `data@2` **det-state chunk-map**
  (`register_state_chunks`, minor 3) — per-chunk `(hash, len)` guest-derived from the layout,
  keyed under the fold domain, with a length-aware host map + det-state-aware covering-span
  verifier and `FamilyRef`/manifest shapes unchanged; the fold pins the ordered chunk hashes and
  the lengths are framing hints, so a lying descriptor fails blake3 re-verification (no new trust
  surface), binding on the restore wave. [SF-R3] ranged `read_into` is surfaced in the SDK for
  the fold windows. §12.14 [SF-5] promotes to normative as implemented (the genesis
  `state_contract` is shipping-consumed; `GuestCfg.init` deleted), and because that surface
  becomes wire-visible here **`VhcProtoVersion` bumps 1 → 2** (§10.3 trigger; app↔node
  `WireVersion` unchanged, no DTO change). No op shape, trap code, or journal kind changes beyond
  the reserved `register_state_chunks` minor-3 entry.
- **Draft 3 — amendment a11 (2026-07-20, the SDK streaming fold engine; label unchanged).**
  SDK-behavior only — no op shape, trap code, journal kind, or grant bound changes. §12.14
  [SF-8]'s pinned schedule and digest carry gain their consuming engine in the SDK profiles
  layer: the flagship profile's ingest and make_update land as completion-driven multi-slice
  walks (window inputs in, emitted family windows out; ABI-agnostic — the guest driver owns the
  fetch/emit wiring), with the digest carry threaded through the ingest walk's emission order
  and the walk refusing mis-geometry (profile-chunk / window-alignment violations) typed at
  construction. §13's det-state bullet promotes the **window ≡ resident fold parity** suite
  from "later waves" to landed for the flagship profile (bit-identical masters, payload
  sections, error-feedback state, and digests across window geometries — including the
  degenerate single-window tier and a ceremony-shaped scaled layout — in-flight bounds, and
  arrival permutations); the resident profile APIs remain the parity oracle, and the remaining
  profiles' windowed forms stay with the resident-path deletion wave (which must window them or
  explicitly retain the resident path with a documented rationale).
- **Draft 3 — amendment a10 (2026-07-20, the det-state write ops + host state store; label
  unchanged).** §12.14's reserved op clauses promote from "shape fixed" to **normative as
  implemented**, shapes unchanged: [SF-4] — the three `vhc@2` state imports
  (`state_open`/`state_emit`/`state_seal`) land at **`vhc@2` minor 3** (the registry entries +
  the `DA_ABI_MINOR_V2` bump ride the same change that wired the host state store into the
  event-loop driver, the §12.7 register_chunks precedent); the seal's nr record is the
  journal-record-only tag-2 **kind 6** (never a callable `read_back` kind, the kind-4/5
  discipline); misframed emits and incomplete seals trap the new typed codes
  `StateMisframedEmit` / `StateIncompleteSeal` (framing is deliberately coarse — `0 < len ≤
  chunk_size` per emit, `Σ len == byte_len` at seal; per-parameter tail alignment is a
  fold-identity concern the parity suites and `expected_root` catch, not a host trap). [SF-R1]
  of [SF-3] — self-sealed folds fetchable by construction, serviced host-locally through the
  ordinary Completion protocol, replayed from a replay-side state chunk store re-executed from
  guest memory (no `ReplayMissingPayload` for self-sealed roots; the journal stays O(records)).
  [SF-7] — the four grant bounds enforced at their declared points (`state-write-budget` at
  emit, `state-store-bytes` at seal after retention eviction with rollback on refusal,
  `state-streams-max` at open, `state_retain_roots` per-family oldest-first eviction with
  checkpoint-pin exemption). Torn folds ([SF-4] crash rule) are host-enforced: only the seal
  mints a durable artifact; unsealed streams force-reclaim at teardown and the store is
  instance-scoped. [SF-R2]/[SF-R3] (externally-sourced roots; ranged `read_into` surfacing) and
  [SF-6] stay reserved for their owning waves.
- **Draft 3 — amendment a9 (2026-07-20, the chunk-addressed det-state contract; label
  unchanged).** New §12.14: the det-state contract — the corpus custody chain instantiated for
  canonical training state under the new derivation domain `daemon-vhc/det-state/1.0.0`
  ([SF-1] family vocabulary, [SF-2] fold identity + per-round manifest + round state root,
  [SF-5] the genesis state contract + its authoring validation rules, [SF-8] the streaming
  digest carry + the slice-decomposition schedule), with the read rules, write ops, checkpoint
  document v2 carriage, and retention bounds shape-fixed and reserved for their implementing
  waves ([SF-3], [SF-4], [SF-6], [SF-7]). §12.7 [CC-2] notes the fold definition is a shared
  chunk-addressed artifact-identity clause instantiated per domain. §13 gains the shared-vector
  conformance suites (chunk-geometry refusals, digest-carry equivalence, fold-walk schedule).
- **Draft 3 — amendment a8 (2026-07-20, registry-served iroh roster; label unchanged).** New
  §12.13: the signed iroh roster record (`daemon-vhc/roster-record/1.0.0`) — a node's
  per-run-key-signed, certificate-carried reachability statement (endpoint id + direct
  addresses + optional relay URL), stored at the registry under a structural
  `(incarnation, issued_at_ms)` monotonic upsert (untrusted storage; peers verify + apply
  freshness precedence per `(role, base identity)`). New registry routes
  `GET`/`PUT {base}/runs/:id/roster`. Additive node↔worker wire: the plane-selection
  credentials' iroh half gains `bind_addr` (the node-pinned socket, so published addresses and
  the bound endpoint agree by construction).
- **Draft 3 — amendment a7 (2026-07-19, execution-backend capability/selection/placement wire;
  label unchanged).** New §12.12: the compute@2 runner executes on the ADMISSION-selected
  backend (the driver's unconditional CPU runner is retired). Additive node↔worker wire:
  `Hardware.backends` carries structured `backend-capability` records (lane slug, device class,
  adapter, device index, vram / per-buffer ceiling / shared MiB, unified, ready — the CUDA
  record's `ready` is the two-leg NVRTC gate), and the admitted tuple gains
  `backend`/`gpu_index`, rederived at join by rerunning the measured selection ladder
  (cuda → wgpu → cpu over the advertised records, filtered by `device_min.backend_class`) —
  the join-time comparison IS the device-claim revalidation. No silent fallback anywhere: an
  admitted-but-unservable backend is the typed `BackendUnavailable` refusal (recoverable —
  `failed_retryable`), deferred device faults at fence/readback are the recoverable CAPACITY
  class, and the per-buffer ceiling is advertised for fleet-preflight sizing rather than
  enforced at admission. Host process discipline recorded: one device-backed compute instance
  per process; GPU backends constructed on and driven from the per-instance guest thread. All
  fields serde-default additive; no existing encoded byte changes; the minor is not raised.
- **Draft 3 — amendment a6 (2026-07-18, wire-schema ownership made normative; label unchanged).**
  New §12.5: the wire surface splits into **mechanism** (owned by `daemon-vhc-proto`: canonical
  CBOR, signing, hashes, merkle, genesis envelope, grants, certificates/revocations, transition
  chain) and **round vocabulary** (owned by the SDK schema layer `daemon-vhc-sdk-consensus`: the
  round messages + `VhcMessage`/`SignedMessage`, the round state-digest schedule, the
  `record-set.cbor` object, and their CDDL — relocated out of the proto with byte-identical CBOR
  encodings; the moved conformance suite pins them). Production host crates are structurally
  barred from linking the schema crates (dependency gate + a negative architecture test on the
  resolved default graph of the shipped worker); the cloud coordinator DO is a module seat whose
  compiled decision core is the same `coordinator::tick` the node's coordinator-quorum guest
  wraps, with a vendored, provenance-recorded TS schema mirror for its I/O edges. §12.5 [OWN-5]
  inventories the two coexisting signed-frame forms (the §12.1 domain-separated envelope vs the
  control-plane `SignedMessage`) as a transitional divergence whose convergence is a
  separately-ratified wire change (a WireVersion-accumulation input). No encoded byte changes;
  the minor is not raised.
- **Draft 3 — amendment a5 (2026-07-18, live-upgrade fence semantics made explicit; label
  unchanged).** Three previously-unstated live-upgrade behaviors are folded in as normative text,
  all consistent with the existing wire (no `frame-envelope`/journal/manifest field added, removed,
  or changed): (1) the journal is **continued in one log** across an upgrade — the retired
  incarnation's records remain as the prefix and the incoming incarnation opens its span at the
  seam with its own tag-0 run-header, the per-journal ordinal staying globally monotone and the
  seam forcing a segment roll (§8.1; restated in §10.3); (2) the node's live-upgrade command is
  named **`switch_module`** (the upgrade-time peer of `AssessRun`/`JoinRun`), and a worker answers
  it with a typed command-unsupported result until its session holds a long-lived run instance to
  migrate (§10.3); (3) across the upgrade fence the per-channel `publish` sequence **restarts at 0**
  in a fresh per-incarnation stream — the upgrade-fence case of §12.2's never-reused stream scope
  (§10.3, cross-referencing §12.2). No wire change at major 2 minor 0; the minor is not raised.
- **Draft 3 — amendment a4 (2026-07-17, hardware conformance: CUDA fence-visibility gap CLOSED +
  genuine driver OOM exercised; label unchanged).** The fence-visibility gap recorded in a2/a3 is
  fixed on the CUDA backend by a vendored single-crate patch of `cubecl-cuda 0.10.0` whose alloc
  path records failed reservations on the stream error queue (like `launch`/`write`) instead of
  `initialize_memory` panicking on `reserve(size).unwrap()`, so `RunnerClient::sync`/`fence`
  drains the typed `ComputeError::Device`/`ComputeFault`. Validated on RTX 4090 (branch
  `vhc/cuda-fence-visibility`, cuda-gated `compute_cuda.rs`): a host-side pool-cap rejection is now
  fence-visible typed, and a **genuine driver `CUDA_ERROR_OUT_OF_MEMORY`** (pool-acceptable
  allocations summing past free VRAM — the driver engaged, unlike the a2/a3 single 2×-VRAM buffer)
  is fence-visible AND readback-visible typed, enqueue infallible, host + context survive
  (non-sticky). An A/B against unpatched cubecl confirms fence returned `Ok` before the patch. The
  §15 hardware-conformance note gains a dated update. Residual: genuinely sticky (illegal-address /
  context-poisoning) semantics stay unvalidated — unreachable through the pinned op set (`Custom`
  refused pre-dispatch). No wire change at major 2 minor 0; the minor is not raised.
- **Draft 3 — amendment a3 (2026-07-16, same-day correction of a2 after an adversarial audit;
  label unchanged).** A second agent re-verified a2's raw logs and code against its claims and
  found the stimulus over-claimed. Corrections, applied to the §15 hardware-conformance note in
  place: the fault was **not** a genuine driver-reported `CUDA_ERROR_OUT_OF_MEMORY` — it was
  cubecl's **host-side `IoError::BufferTooBig`** pool-cap rejection of a 2×-VRAM (50.5 GB)
  single-buffer request (no memory pool's `max_alloc_size` accepts it,
  `cubecl-runtime` `memory_manage.rs:496-503`), refused before any `cuMemAlloc`; the driver was
  never engaged (`CUDA_ERROR_OUT_OF_MEMORY` appears in zero logs). "The device stream died" was
  inaccurate — cubecl catches the panic per-task and the same stream/runner keeps serving; the
  fault is scoped to the one unbacked handle. "Non-sticky" was misleading — no driver-level
  error existed to be sticky. The root cause of the fence gap is not `RunnerClient::sync`
  failing to observe stream health (the sync plumbing would surface a `ServerError` if one
  existed) but `cubecl-cuda`'s `initialize_memory` doing `command.reserve(size).unwrap()`
  (`cubecl-cuda` `server.rs:124`) — panicking on the fallible alloc instead of pushing the
  error into the stream's error queue as the `launch` path does (`server.rs:157-163`); the
  caught-and-dropped panic leaves no error state for sync to drain, so the fix is to make the
  alloc path record errors like the launch path. What a2 claimed that **stands** (audited
  against the raw logs): infallible enqueue, the typed `ComputeFault` readback surfacing (a
  generic fault for the affected handle, not an OOM diagnostic), host survival, same- and
  fresh-runner recovery, `fence()` Ok after the fault, and the normative rule (fence-visibility
  REQUIRED but unmet; hosts MUST NOT infer device health from fence success; readback
  authoritative in the interim). Residual is **broader** than a2 recorded: true driver-reported
  OOM **and** sticky-error semantics both remain unvalidated — the spike's escalated
  deferred-error gap stays open; only the host-side pool-cap rejection case is closed. No wire
  change; the minor is not raised.
- **Draft 3 — amendment a2 (2026-07-16, hardware conformance: CUDA fence-visibility gap
  recorded; label unchanged; corrected same day — see a3).** First genuine-device-fault validation of the ratified
  `compute@2` deferred-error rule on real hardware (RTX 4090, remote CUDA run; vhc-integration
  `9a7185c3`, cuda-gated evidence test `compute_cuda.rs` landed at `d856538e`). Typed readback
  surfacing, host survival, OOM non-stickiness, and cuda-tier bit-exact digest reproduction
  (`v2_parity`, determinism suites) all held; but `fence()` returns Ok after a device-stream
  death — burn/cubecl's `RunnerClient::sync` does not observe stream health, so deferred faults
  are readback-visible only. §15's `compute@2` reserve gains a normative hardware-conformance
  note: fence-visibility stays REQUIRED; hosts must not read device health from a successful
  fence until the cubecl sync bridging lands (tracked post-Phase-E). No wire change at major 2
  minor 0; the minor is not raised.
- **Draft 3 — amendment a1 (2026-07-15, post-spike: ratified `compute@2` direction recorded;
  label unchanged).** The Burn-over-`HostBackend` prototype spike — the Phase-C entry gate this
  document defers to (§15, §16) — passed (verdict PROCEED with amendments; tier-1 bit-exact), and
  the human ratified its reframing as decisions **D8**. This pass records the ratified direction
  inside the §15 `compute@2` reserve: the wire is CBOR-encoded `burn_ir::OperationIr` at the
  pinned Burn version per ABI major (`burn = 0.21.0`; a Burn bump is an ABI event); opaque
  `TensorId` handles, guest-side metadata/refcount/`Drop`, deferred errors at fence/readback,
  stale handle = typed trap (mapping specified in C); quantization/`QFloat` and custom-op IR
  variants RESERVED (clean refusal until specified); no autodiff ABI surface —
  `backward@1`/`grad@1` retire under `compute@2`. The surface **stays reserved** (nothing links
  before Phase C); §16's deferred-items note is updated to match. No wire change at major 2 minor
  0; the minor is not raised.
- **Draft 3 — erratum e2 (final pre-ratification alignment pass; label unchanged).** Records the
  five surgical corrections applied under the ratification review's conditional approval. This
  companion is itself touched only at §2.6 and here: the §2.6 grants-document CDDL comment on
  `instance` is corrected from "role-instance ordinal" to the never-reused durable u64 role-instance
  incarnation (matching erratum e1's §8.1/§12.2 semantics; the CDDL `instance: uint` is unchanged).
  The companion edits this pass makes elsewhere: **architecture §5.1** now describes `instance` as
  the never-reused durable u64 incarnation (not an ordinal) — correcting the Draft-3 overstatement
  above that §5.1 was already aligned — and its `(identity, seq)` dedup/gap/evidence wording is
  updated to the full channel-scoped scope `(run_id, epoch, role, incarnation, channel, seq)` with
  the signing-oracle description speaking in envelope/driver-declared channels (host derives
  class/routing/bounds) rather than a guest-selected message class (§12.1/§12.2/§6.2 unchanged);
  and **refactor §4/§5** reword driver selection to the §1.3 order, restate deliverable 6 as
  per-device + host-wide typed ledgers (decisions D6), correct the §9 identity line, and pin the
  A0 frozen-fixture definition and the A1/A0/A2 launch boundaries. The **decisions doc's
  ratification checklist is marked RATIFIED** under this conditional approval. No wire change; the
  minor is not raised.
- **Draft 3 — erratum e1 (decisions-doc audit alignment; label unchanged).** §8.1 and §12.2 are
  clarified surgically: the execution-identity `instance` is a **never-reused, node-durable,
  monotonic `u64` role-instance incarnation id**, explicitly *not* a reusable small ordinal/`u16`
  slot — a reusable slot value would let a fresh role-instance inherit or collide with a retired
  incarnation's durable channel-scoped sequence counter (§12.2), which is unsafe for journals,
  sidecar ownership, and equivocation evidence. This is a wording/semantics clarification of an
  already-normative field (the CDDL `instance: uint` is unchanged), adopted from the daemon-vhc
  decisions-doc audit (decisions D1); it does not change the wire and does not raise the minor.
- **Draft 3 (this revision).** Incorporates the adopted second ratification review (nine
  amendments): a canonical top-level **grants document** with derivation, Phase-A form, and
  assess→join hash-pinning (§2.6); the ABI's `ParticipationLane` schema declared the single
  canonical one (raw bytes/bps units, `bridge_allowed`, `replay_window`, claim-tier →
  admission/arbitration mapping) with the decisions doc re-pointed at it (§9.6, §9.1); workable
  migration bootstrap mechanics (`stage_state` import, migration descriptor with direct staging
  IDs, `read_back(state-section)` legal during `da_migrate`, instantiation journaled before
  init/migrate, retryable snapshot submission — §10.2, §6.4, §6.6); one machine-valid journal
  CDDL grammar validated in tier-1 CI, seal-hash self-reference excluded (§8.3, §8.2); the
  minimum domain-separated signed-frame schema lands at **A2** (D1 adds keys/Authority only)
  (§12); an exhaustive **bridge replay registry** classifying all 66 `tabi@1` imports (§2.7);
  the sidecar AEAD profile (XChaCha20-Poly1305) and a normative logical-time contract
  (slice-constant, journaled with each event) (§8.5, §6.5); the execution identity frozen as
  `(run_id, epoch, role, instance, module_hash)` and applied to journal/sidecar/signing/sequence
  scopes (§8.1, §12); Cell 5 made conditional on a three-proviso envelope fixture with the
  device-minimums CDDL defined (§9.3). Companion docs aligned: decisions D1/D3/D6/D7,
  refactor §5 A2 + §9. (This entry also listed **architecture §5.1** as aligned; that was an
  overstatement — §5.1's execution-identity `instance` wording was *not* corrected in this
  revision. Erratum e2 below is the pass that actually aligns architecture §5.1.)
- **Draft 2.** Incorporates the adopted third-party ratification review:
  compile-before-`da_abi` driver selection via static-import inspection with an export cross-check
  (§1.3); the two-instance assessment/run model with deny-on-call stubs (§9.2); the explicit
  12-step admission/bootstrap state machine and the new required `da_init` export (§9.4, §2.1);
  the transitional `tabi@1` compute bridge under major 2 through Phase C (§2.5); guest-provided
  `next_event` buffers with `NeedCapacity` (§4.1); `Quiesce` split from `Stop` (§4.4);
  manifest-based migration replacing byte-slice state transfer (§10.2); the three-tier resource
  model with replay-deterministic generations (§7.1); channel-scoped sequence numbers (§12);
  channel-based `publish` with a Phase-A default channel table (§6.2). All wire types now have
  complete CDDL (manifest/grants §2.3, events §4, journal §8.3, read_back §6.4, completions §7.5).
  Journal durability (fdatasync barriers, atomic publish), signed-frame evidence, encrypted
  readback sidecars, and replay-failure semantics are specified (§8.4–§8.7). Unknown event tags
  now fail closed (§5.2). Former open questions OQ-1…OQ-15 are resolved into the body; §16 is now
  the resolution log.
- **Draft 1.** Initial draft.

---

## 1. ABI version negotiation and driver selection

### 1.1 The `da_abi` export (normative)

Every module MUST export:

```
da_abi() -> u32          // (major << 16) | minor
```

The return packs a 16-bit major and 16-bit minor, exactly as the retained v1 gate does today
(`daemon-train-sdk::DA_ABI_VERSION = (major << 16) | minor`; host decode `(v >> 16, v & 0xffff)`).
`da_abi` MUST be callable immediately after instantiation with no prior call, MUST be pure (no
imports, no state), and MUST return a compile-time constant.

- `major` names the **driver** the module was built for (§1.2).
- `minor` advertises the core additive capability level *within* a major (§1.4).

`da_abi` is a **declaration to be cross-checked**, not the selection input: the host selects the
candidate driver from the module's *static import namespaces* (§1.3) — it cannot call `da_abi`
before it has linked and instantiated, and linking already requires choosing a linker.

### 1.2 The major gate is driver selection

The `da_abi` **major** is a driver selector, not merely a compatibility check. The one worker binary
carries two drivers for the whole transition (refactor §4 decision 2, §5 A0):

| `major` | Driver | Entry shape | Admission input |
|---|---|---|---|
| `1` | **Retained v1 five-phase driver** | `da_build` → `da_step`/`da_inner_update`/`da_make_update`/`da_ingest_updates`, host-sequenced; phase-legality table enforced | autotune/`MetaReport` probe |
| `2` | **Event loop** | `da_init` + `da_run` + blocking `next_event` (§3–§4); module owns its loop | `claim()` (§9) |
| other | — | refused | typed refusal (§1.5) |

The v1 driver, its `tabi@1` import vocabulary (66 imports, frozen), the phase-legality table, and
autotune-based admission are **frozen and retained unchanged** until the Phase E sunset whose
criteria Phase 0.5 fixed (refactor §4 decision 5, §9, invariant 2). This document does not restate
the v1 contract; it is the existing `swarm-tensor-abi-spec`/`tabi@1` surface, pinned by the A0
fixture. This document defines **major 2**. Note that major-2 modules MAY *additionally* link the
frozen `tabi@1` vocabulary as a transitional compute bridge through Phase C (§2.5); the presence of
`tabi@1` imports alone does NOT make a module major-1.

A host MAY implement only a subset of majors (e.g. a future host that has sunset v1 implements only
major 2). The set of majors a host implements is a static host property, reported to the node in the
worker capability vocabulary (§1.6).

### 1.3 Admission-time ABI selection (normative)

The host MUST perform selection in this order. No union linker is used, and no wasm custom section
is consulted; the selectors are the module hash, the static import section, and the exports.

1. **Verify before compiling.** The module blob's blake3 MUST be verified against the envelope's pin
   for this role (architecture §5.2) **before** the blob is passed to the compiler. A mismatch is
   `ModuleHashMismatch` — an admission fault, and no byte of the blob reaches wasmtime.
2. **Compile and inspect.** Compile (validate) the module. Read its **static import section** and
   export list. Select the **candidate driver**:
   - imports contain any symbol in the `vhc@2` namespace → candidate **major 2**;
   - otherwise, imports are within `tabi@1` and the exports include the v1 lifecycle (`da_build` …)
     → candidate **major 1**;
   - otherwise → `BadModule` (no recognizable driver shape).
3. **Derive and validate the compatibility tuple** (§1.4). From the import section, compute
   `{core_minor_required, world → minor_required, bridge: bool}` — the highest minor at which each
   imported symbol was introduced, per namespace, per the contract crate's symbol registry. Every
   imported symbol MUST be one the host provides at a version ≥ required, else refuse:
   `WorldMinorUnsupported` naming the namespace and symbol (this subsumes "missing import" for
   known namespaces; a wholly unknown namespace is `BadModule`). If the tuple includes
   `bridge = true` on a host that has retired the bridge (§2.5), refuse `BridgeRetired`.
4. **Instantiate** with the linker matching the candidate. For candidate major 2 the first
   instantiation is the **assessment instance** with deny-on-call stubs (§9.2) — capability imports
   resolve at link time but trap `ClaimCapabilityDenied` if called.
5. **Cross-check the declaration.** Call `da_abi()`. Decode `(major, minor)`:
   - `major` ≠ the candidate selected in step 2 → refuse `AbiDeclarationMismatch` (the module's
     declaration contradicts its own import shape);
   - `major` not implemented by this host → `AbiUnsupportedMajor`;
   - `minor > host_minor_for(major)` → `AbiMinorTooNew`;
   - `minor < core_minor_required` from step 3 (the module *imports* symbols newer than it
     *declares*) → `AbiDeclarationMismatch`.
   A module MAY declare a `minor` lower than the host's; the host MUST admit it and MUST NOT
   deliver events or trailing fields above the declared minor (§1.4, §5.2).
6. **Check required exports** for the selected major (§2.1) with correct signatures; a
   missing/mis-typed required export is `BadModule`.

Only after 1–6 succeed does admission proceed to `da_manifest` and `da_claim` in the assessment
instance (§9.4 steps 4–7). The run instance is a **separate, later instantiation** (§9.2).

### 1.4 The compatibility tuple and additive growth (normative)

A module's negotiated compatibility is the tuple:

```
compat = { core: (major, minor), worlds: { "net" → minor, "sys" → minor, … }, bridge: bool }
```

derived from static imports (§1.3 step 3), cross-checked against `da_abi` (core) and the manifest's
declared worlds (§2.3): **static imports ⊆ manifest-declared ⊆ granted ⊆ host-advertised**, each a
typed refusal when violated (`AbiDeclarationMismatch` for imports ⊄ manifest; `GrantsExceedLane` for
manifest ⊄ grants).

Within a major, growth is **additive only** and monotone in `minor`:

- A new minor MAY add import symbols, event variants (§5.2), optional manifest/claim fields, and
  completion result variants (§7.5). It MUST NOT change the signature or semantics of any symbol
  defined at a lower minor, remove any symbol, or renumber any wire tag or enum value.
- A module built at minor *m* MUST run unchanged on any host at minor ≥ *m* within the same major.
- The host MUST NOT deliver an event variant or trailing field introduced at minor *m* to a module
  that declared `minor < m`; per-world minors gate world-specific variants identically. §5.2 makes
  violation of this rule fail closed.
- New optional CBOR map fields use the existing additive discipline (absent field decodes with a
  default on an older reader), exactly as the v1 `Manifest::max_round_interval_ms` and
  `Hardware::shared_mb` fields already demonstrate.

**Exclusion:** the `tabi@1` bridge under major 2 (§2.5) is a declared-transitional surface
explicitly **excluded** from the additive guarantee; its availability is the `bridge` flag in the
host's advertised tuple and its retirement is governed by the Phase C gate (§2.5).

Anything that would change control-flow shape (a new blocking import, a change to `da_run`'s
contract, a change to delivery ordering) is a **new major**, with the lower major retained as a
compatibility driver (architecture §10).

### 1.5 Typed refusal errors (normative)

Version-negotiation and admission failures are **admission refusals**, surfaced to the node as typed
`AssessRun`/instantiate outcomes — never wasm traps and never worker crashes. Internally every
refusal belongs to the broad category **`AbiMismatch`** (retained as the v1 code and as the umbrella
for coarse consumers); the exposed surface carries the split code (resolution of OQ-1):

| Exposed code (slug) | Meaning | Raised at (§1.3/§9.4 step) |
|---|---|---|
| `ModuleHashMismatch` | blob blake3 ≠ envelope pin | 1 |
| `BadModule` | unrecognizable driver shape, unknown import namespace, missing/mis-typed required export | 2, 6 |
| `WorldMinorUnsupported` | an imported symbol needs a namespace minor this host lacks | 3 |
| `BridgeRetired` | module imports `tabi@1` under major 2 on a bridge-retired host | 3 |
| `AbiUnsupportedMajor` | declared `major` not implemented by this host | 5 |
| `AbiMinorTooNew` | declared `minor` exceeds the host's for that major | 5 |
| `AbiDeclarationMismatch` | `da_abi` contradicts the import-derived candidate/tuple, or imports ⊄ manifest | 5, §9.4-6 |
| `GrantsExceedLane` | manifest requires worlds/ops/channels/depths beyond grants or lane bounds | §9.4-6 |
| `ClaimExceedsPolicy` | claim exceeds lane claim-bounds or owner resource authorization | §9.4-7/8 |
| `ClaimInconsistent` | repeated `da_claim` invocations returned different bytes | §9.4-7 |
| `MigrateUnsupported` | `switch_module` targets a module without `da_migrate` — **always an admission refusal, never a trap** | upgrade re-admission (§10.3) |

Refusals MUST carry `{code, detail}` where `detail` is human-readable and names the offending value
(observed vs supported). They MUST NOT execute any `da_init`/`da_run`/`da_step` guest code.

### 1.6 Mixed-fleet reporting

The host reports, per worker, its full advertised tuple — `{ majors: [u32], minor_by_major:
{major → minor}, worlds: {name → minor}, bridge: bool, custom_ops: [tstr] }` — in the capability
vocabulary (extending `WorkerCapabilities`). This is what the node uses to populate the Phase 0.5
mixed-fleet compatibility matrix ({v1,v2}×{native,wasm coordinator}×{envelope v1,v2}). The matrix
itself is a Phase 0.5 deliverable, not part of this ABI, but every cell's *refusal behavior* is
governed by §1.5.

---

## 2. Module layout — exports and import namespaces (major 2)

### 2.1 Required exports

A conforming major-2 module MUST export **at a minimum** the following (signatures in wasm core
types; the SDK `main!` macro emits them — §10). Additional exports are permitted and ignored by the
host.

```
da_abi()                              -> u32   // §1.1  (major<<16)|minor
da_alloc(size: u32, align: u32)       -> u32   // §2.4  guest buffer for host writes (outside imports)
da_free(ptr: u32, size: u32, align: u32)       // §2.4  paired release
da_manifest(cfg_ptr: u32, cfg_len: u32) -> u64 // §2.3  (ptr<<32)|len canonical CBOR
da_claim(cfg_ptr: u32, cfg_len: u32,
         grants_ptr: u32, grants_len: u32) -> u64  // §9  (ptr<<32)|len canonical CBOR
da_init(cfg_ptr: u32, cfg_len: u32,
        grants_ptr: u32, grants_len: u32) -> u32   // §9.4 step 11: initialize with the ADMITTED
                                                   //      config+grants; returns init status
da_run()                              -> u32   // §4    the module main loop; returns Outcome code
```

`da_init` (new in Draft 2, resolution of blocking fix 3) is called exactly once, on the **run
instance** only, before `da_run`, with byte-identical copies of the config and grants that were
admitted (§9.4). It is where the module constructs its state and — under the `tabi@1` bridge — where
registration imports are legal (§2.5). Return: `0` = ready; any nonzero value is a typed init
failure (journaled; the host refuses the join / leaves the run; values ≥ 16 are module-defined and
carried verbatim in the refusal detail).

`da_defaults() -> u64` is **OPTIONAL** and carried over from v1 verbatim (the `[experiment.config]`
defaults layer). If absent the host treats defaults as the empty CBOR map.

`da_migrate(descriptor_ptr: u32, descriptor_len: u32) -> u32` is **OPTIONAL** at Phase A and
REQUIRED for any module that participates in a live upgrade. It receives a canonical-CBOR
**migration descriptor** (§10.2) — never a raw state byte-slice. A module that omits it declares
itself non-migratable; the
host refuses any `switch_module` targeting it with the **admission refusal** `MigrateUnsupported`
(§1.5 — never a trap) and the run falls back to leave-and-rejoin (architecture §5.4).

The `(ptr << 32) | len` packed return convention for CBOR-returning exports is **retained from v1**
(`da_manifest`/`da_defaults`/`da_claim`; SDK `rt::emit_cbor`). The host reads the span, copies it
out, then calls `da_free(ptr, len, 1)`. A `0` return means an empty result.

### 2.2 Import namespaces

Resolution of OQ-2: **loop mechanics live in `vhc@2`, network routing in `net@2`, clock/timers and
ambient telemetry in `sys@2`.** Each namespace is versioned independently (the per-world minors of
§1.4); symbols follow the `tabi@1` naming discipline (`#[link(wasm_import_module = "…")]`,
per-symbol `name@version` link names).

| Namespace | Phase A symbols | Later phases |
|---|---|---|
| `vhc@2` | `next_event`, `read_back`, `stage_state` (§10.2), `snapshot_state` (quiesce-scoped, §10.2) | `cancel` (Phase B completions); `state_open`/`state_emit`/`state_seal` (the det-state write surface, minor 3 — §12.14 [SF-4]) |
| `net@2` | `publish` | payload put/get, gossip, streams, credit (Phase B) |
| `sys@2` | `set_timer`, `cancel_timer`, `now`, `emit_metric`, `log` | seeded RNG, device profile, crypto accelerations (Phase B) |
| `data@2` | — (none at Phase A) | artifact fetch by hash (Phase B); chunk-map registration for chunk-addressed corpus shards, minor 2 (§12.7) |
| `compute@2` | — (none at Phase A) | Burn-shaped surface, fences (Phase C) |
| `tabi@1` | the frozen 66-import v1 vocabulary, as the transitional compute bridge (§2.5) | retires at Phase C |

### 2.3 The manifest (`da_manifest`)

`da_manifest(cfg)` returns canonical CBOR describing the module's static requirements — the
admission contract (architecture §3.2, §5.2: `required ⊆ granted ⊆ advertised`). It is a **pure
function of the config**, MUST NOT call any capability import (the assessment instance's stubs trap
`ClaimCapabilityDenied` if it tries, §9.2), and MUST be byte-identical across repeated invocations
with the same config (else `ClaimInconsistent`, §9.4).

```cddl
manifest = {
  "name":       tstr,
  "version":    tstr,
  "sdk":        tstr,
  "abi":        uint,                ; da_abi() echoed, cross-checked (§9.4 step 6)
  "worlds":     [* world-req],       ; every namespace the module statically imports MUST appear
  "custom_ops": [* tstr],            ; versioned host custom-op names (architecture §3.2)
  "channels":   [* uint],            ; channel IDs the module will publish/subscribe (§6.2)
  "events":     event-caps,          ; advisory-class queue depth + coalescing declaration
  "buffers":    buffer-req,          ; live resource quotas (§7)
  ? "migratable": bool,              ; default false; MUST be true iff da_migrate is exported
}

world-req = {
  "world":  tstr,                    ; "vhc" / "net" / "sys" / "data" / "compute" / "tabi"
  "minor":  uint,                    ; namespace minor required (for "tabi": always 1)
  "grants": { * tstr => grant-bound },  ; requested per-grant bounds
}

grant-bound = {
  ? "max_bytes":       uint,         ; per-item byte ceiling (e.g. publish payload, readback value)
  ? "max_per_slice":   uint,         ; per-event-slice call ceiling for this grant
  ? "rate_per_min":    uint,         ; sustained rate ceiling (token bucket, per minute)
  ? "max_outstanding": uint,         ; concurrent-operation ceiling (Phase B completions)
  ? "values":          [* tstr],     ; enumerated allowed values (topics, dataset hashes, sources)
}
; Absent keys mean "unbounded by this grant" — still bounded by the lane ceiling (§9.6).

event-caps = {
  ; advisory classes the module subscribes to, with the declared depth + coalescing rule the host
  ; checks at admission and enforces at runtime (§5.4). Authoritative channels are NOT declared
  ; here — their bounds come from the channel table (§6.2).
  * tstr => { "depth": uint, "coalesce": uint }
  ; class keys: "payload-ready" / "timer" / "gossip"
  ; coalesce: 0 = dedup-by-hash, 1 = latest-wins, 2 = drop-oldest (fixed per class, §4.7)
}

buffer-req = {
  "max_live_handles":   uint,        ; standing live-resource ceiling (all instance-class resources)
  "max_live_bytes":     uint,        ; standing live-buffer byte ceiling (Phase B buffers)
  "max_readback_bytes": uint,        ; per-slice ceiling on bytes crossing into linear memory
}
```

The host checks `manifest.worlds`/`custom_ops`/`channels`/`events`/`buffers` against the role's
grants and the selected lane's bounds at admission (§9.4 step 6). A manifest requiring more than the
grants allow → `GrantsExceedLane`; static imports not covered by `manifest.worlds` →
`AbiDeclarationMismatch`. The manifest bytes are journaled in the run header (§8.3 tag 0).

### 2.4 Guest allocator and non-reentrancy

`da_alloc`/`da_free` are retained from v1 (`rt.rs`) with one **narrowed obligation**: the host uses
them ONLY outside guest import context —

- to write the config/grants spans before calling `da_manifest`/`da_claim`/`da_init`, and
- to release spans the guest returned from `da_manifest`/`da_claim`/`da_defaults`.

**Non-reentrancy (normative, resolution of blocking fix 5):** the host MUST NOT invoke any guest
export (including `da_alloc`/`da_free`) while the guest thread is executing inside a host import.
There are no re-entrant guest calls. Consequently `next_event` and `read_back` write only into
guest-provided buffers (§4.1, §6.4); the host never allocates guest memory mid-import.

Allocator semantics (unchanged): `da_alloc(size, align)` returns a linear-memory offset for a
`size`-byte, `align`-aligned region, or `0` for `size == 0`; a `0`/misaligned return for a nonzero
request is `AllocFail` (§7.6). The host MUST call `da_free(ptr, size, align)` with the exact triple
once done.

### 2.5 The transitional `tabi@1` compute bridge (normative; retires at Phase C)

Resolution of blocking fix 4. At Phase A the v2 driver replaces the **lifecycle**, not the **tensor
ops**: a major-2 module MAY statically import the frozen `tabi@1` vocabulary (all 66 imports,
unchanged names, signatures, and semantics) alongside the v2 namespaces, and the host MUST link them
for major-2 modules while the bridge is advertised (`bridge: true` in the tuple, §1.4).

**Rationale (recorded):** this keeps the refactor's A2 acceptance intact — TinyLlama on
`BarrierRound` reproducing v1 det-lane state digests across backends. Digest parity is the evidence
that the control inversion did not change the math; that evidence is only obtainable if the math
runs on the identical frozen op surface while the loop around it changes.

Bridge rules:

1. **Namespacing.** The bridge is the literal `tabi@1` wasm import module — byte-identical link
   names, no re-versioning. The v1 host dispatch layer serves it; the v1 **phase-legality table does
   NOT apply** (there are no phases). In its place:
   - registration imports (`param@1`, `persistent@1`, `det_persistent@1`) are legal **only during
     `da_init`** (the v2 analogue of `da_build`); elsewhere they trap `PhaseViolation`;
   - every other `tabi@1` import is legal in any event slice during `da_run`;
   - no `tabi@1` import is linked in the assessment instance (deny stubs, §9.2).
2. **Staging integration.** The v1 host-pushed staging entries (`da_step`'s batch handle,
   `da_ingest_updates`' staged set) do not exist under v2. Instead the session stages content and
   announces it with `PayloadReady` events carrying a `staging_id`; the guest acquires bridge
   handles via `read_back` kinds 1 and 2 (§6.4): a staged **batch** yields a `tabi@1` batch handle
   (kind 7, deterministic index-based, per `handle.rs`), a staged **update container** yields the
   staging index consumed by `upd_sections@1`/`upd_kind@1`/`upd_read_bytes@1`/`upd_tensor@1`.
   Acquired batches and staged update containers are **instance-class resources** (§7.1): they
   survive slice boundaries until released with `drop@1` or invalidated by restart.
3. **Journal treatment.** Governed exhaustively by the **bridge replay registry** (§2.7): every
   `tabi@1` import is classified there, and every import whose result bytes can influence guest
   control flow or outbound bytes is either det-lane or recorded verbatim as a `read-back`-class
   journal record (§8.3 tag 2) with a reserved kind ≥ 128. Tensor *contents* never enter the
   journal (staged inputs are hash-pinned; native tensor state reaches the guest only through
   registry-recorded readouts).
4. **Budget accounting.** Every bridge call counts against the per-slice op budget and fuel exactly
   like a v2 import (§5.5); bridge handle classes obey the three-tier resource model (§7.1): v1
   step tensors are slice-class, params/persistents/det-persistents are registered-class
   (registered in `da_init`), batches/update containers are instance-class.
5. **Retirement.** The bridge retires **at Phase C**: when the host ships `compute@2`, the bridge
   flag flips off under the same fixture-gated discipline as the v1 sunset (all major-2 guests
   rebuilt against `compute@2`; one full release cycle of dual support; owner notice). A module
   importing `tabi@1` under major 2 on a bridge-retired host is refused `BridgeRetired` at §1.3
   step 3 — a typed admission refusal, never a runtime surprise. Because the bridge is excluded
   from the additive-minor guarantee (§1.4), this retirement is not a compatibility-contract
   violation.

### 2.6 The grants document (normative)

The **grants document** is the single canonical value naming everything a role-instance may reach.
One byte string serves every consumer: it is passed to `da_claim` and `da_init` (§2.1), journaled
verbatim in the run header (§8.3 tag 0 `grants`), re-checked at upgrade re-admission (§10.3 step 3),
and consulted for migration restore authorization (§10.2). It is canonical CBOR; its blake3 is the
**grants hash** used for assess→join pinning (below) and the tag-11 `da_init` cross-check.

```cddl
grants-doc = {
  "version":      uint,               ; grants-schema version; 1 in this document
  "run_id":       bstr .size 32,
  "epoch":        uint,
  "role":         tstr,
  "instance":     uint,               ; role-instance incarnation — never-reused durable u64 (§8.1 execution identity, erratum e1)
  "lane":         tstr,
  "lane_version": uint,
  "worlds":       { * tstr => world-grant },
  "custom_ops":   [* tstr],
  "channels":     [* channel-decl],   ; the ADMITTED channel table (§6.2)
  "events":       event-caps,         ; admitted advisory depths (≤ manifest request)
  "buffers":      buffer-req,         ; admitted quotas (≤ manifest request)
  ? "migration":  migration-grant,
}

world-grant     = { "minor": uint, "bounds": { * tstr => grant-bound } }
migration-grant = { "restore": bool, "max_sections": uint, "max_section_bytes": uint }
```

(`channel-decl` §6.2; `event-caps`/`buffer-req`/`grant-bound` §2.3.)

**Derivation (normative).** The host computes the admitted grants as the pointwise intersection

```
admitted = lane profile ceilings (§9.6)  ∩  envelope role grant list  ∩  owner standing policy
```

taking, per bound, the tightest value (minimum of numeric ceilings, intersection of enumerated
sets, conjunction of booleans), then intersecting the result with the module's manifest *requests*
(§2.3) — an admitted value is never looser than any contributor and never grants what the manifest
did not request. The result is serialized once, canonically, at admission; every later consumer
receives those exact bytes.

**Phase-A form (envelope v1).** Until D0 there is no envelope role grant list. The Phase-A
contributors are: the lane profile (Trainer, the only enabled lane), the owner's standing policy,
the **driver-provided default channel table** (§6.2) as the `channels` value (with
`max_frame_bytes` additionally tightened by the v1 envelope's `[requirements].update_mb_max` where
present), and the manifest's requests. `run_id` is the frozen v1 envelope hash; `epoch = 0`.

**Assess→join byte identity (normative).** Assessment (§9.4 steps 1–8) pins
`blake3(config bytes)` and `blake3(grants bytes)` together with the claim result. On `JoinRun` the
host re-runs **only** the cheap owner-authorization stage (§9.4 step 8) against the recorded claim;
if the pinned hashes still match the freshly re-derived config/grants (i.e. envelope, lane, and
owner policy are unchanged), it proceeds directly to instantiation, and `da_init` receives
byte-identical copies verified against the pinned hashes (§8.3 tag 11). On any hash mismatch the
host re-runs §9.4 steps 1–8 in full — it never joins on stale admission. **There is deliberately no
signed assessment token**: assess and join both execute inside the same node client's trust domain,
so a token would authenticate a channel that has no adversary — hash-pinning gives the byte-identity
guarantee without inventing a signer (rationale recorded per ratification).

### 2.7 The bridge replay registry (normative)

Input replay claims that re-feeding a journal reproduces every decision **without re-running native
kernels** (§8.7). Under the bridge that claim is only a theorem if every one of the 66 `tabi@1`
imports (the frozen `TABI_IMPORTS` list, `daemon-train-sdk/src/lib.rs`) is classified by how its
guest-visible result is reproduced at replay. This registry is that classification; it is exhaustive
and closed (the bridge surface is frozen), and the conformance suite asserts it covers
`TABI_IMPORTS` exactly.

**Classes and their replay treatment:**

- **dc — deterministic-control.** The result (a handle, or nothing) is a deterministic function of
  the handle-table/registration bookkeeping (§7.1), which the replay verifier re-executes; the
  native kernel body is **skipped**. Call + arguments are asserted.
- **dd — deterministic-det-computation.** Det-lane ops: same replay treatment as dc at input
  replay (handles from bookkeeping, bodies skipped), with the additional normative property that
  their *semantics* are bit-exact per `daemon-vhc-det`, so consensus replay (architecture §3.6
  tier 3) can re-derive their state from committed inputs.
- **nr — nondeterministic-result-recorded-verbatim.** The result bytes are journaled as a
  `read-back`-class record (§8.3 tag 2) with the reserved kind below; replay feeds the recorded
  value. Sidecar rules apply above `READBACK_INLINE_MAX` (§8.5).
- **se — side-effect-asserted-during-replay.** No guest-visible result; the effect is on host
  state (device tensors, container build, telemetry). The body is skipped; the replay verifier
  asserts the call occurred with identical arguments.

| Import | Class | Import | Class | Import | Class |
|---|---|---|---|---|---|
| `param@1` | dc | `adamw_step@1` | se | `det_add@1` | dd |
| `persistent@1` | dc | `batch_tokens@1` | dc | `det_sub@1` | dd |
| `det_persistent@1` | dc | `batch_size@1` | **nr** (130) | `det_mul@1` | dd |
| `drop@1` | dc | `batch_seq_len@1` | **nr** (131) | `det_absmax_unpack@1` | dd |
| `param_round_base@1` | dc | `upd_new@1` | dc | `det_chunk_scatter_add@1` | dd |
| `backward@1` | se | `upd_push_bytes@1` | se | `det_chunk_scatter@1` | dd |
| `grad@1` | dc | `upd_push_tensor@1` | se | `det_assign@1` | dd |
| `zero_grads@1` | se | `upd_sections@1` | **nr** (132) | `det_param@1` | dd |
| `assign@1` | se | `upd_kind@1` | **nr** (133) | `det_reset_param_to_base@1` | dd |
| `zeros@1` | dc | `upd_bytes_len@1` | **nr** (134) | `det_axpy_param@1` | dd |
| `ones@1` | dc | `upd_read_bytes@1` | **nr** (135) | `embedding@1` | dc |
| `full@1` | dc | `upd_tensor@1` | dc | `rmsnorm@1` | dc |
| `add@1` | dc | `det_zeros@1` | dd | `softmax@1` | dc |
| `sub@1` | dc | `det_sum@1` | dd | `silu@1` | dc |
| `mul@1` | dc | `det_scale@1` | dd | `rope@1` | dc |
| `mul_s@1` | dc | `det_l2norm@1` | **nr** (136) | `flash_attn@1` | dc |
| `matmul@1` | dc | `det_sign@1` | dd | `reshape@1` | dc |
| `relu@1` | dc | `scalar@1` | **nr** (128) | `transpose@1` | dc |
| `cross_entropy@1` | dc | `metric@1` | se | `slice@1` | dc |
| `abi_minor@1` | **nr** (129) | `log@1` | se | `topk_chunk@1` | dc |
| `chunk_scatter@1` | dc | `absmax_pack@1` | dc | `absmax_unpack@1` | dc |
| `dct2@1` | dc | `idct2@1` | dc | `det_idct2@1` | dd |

(66 imports: 34 dc, 15 dd, 9 nr, 8 se. Parenthesized numbers are the reserved §8.3-tag-2 journal
kinds.)

**Registry notes (normative):**

- **`det_l2norm@1` is nr, not dd**, despite being det-defined: its f64 return is bit-exact *given
  det state*, but det state's lineage crosses the native lane (`det_param@1`,
  `det_reset_param_to_base@1`, `det_axpy_param@1` read/write the round-base master, which native
  kernels trained). Input replay does not hold native state, so the value is recorded verbatim;
  consensus replay may still re-derive it where the det state is a pure function of committed
  inputs.
- **`topk_chunk@1`** writes its indices *handle* to a guest out-param — a host write of a
  deterministic (bookkeeping-derived) value into linear memory; the replay verifier reproduces it
  from the arena, so the import stays dc.
- **`upd_read_bytes@1`** can return large sections; it is the one bridge import expected to hit
  the sidecar path routinely.
- **`abi_minor@1`** is a host property, not guest-derivable — recorded (kind 129) so a journal
  replays identically on a verifier at a different host minor.

**The replay theorem (normative claim of this registry).** Every guest-visible result byte under
the bridge comes from a dc/dd handle (reproduced by re-executing only the deterministic
bookkeeping) or an nr record (fed verbatim). Therefore input replay reproduces the guest's complete
decision sequence — including every `publish` payload authored from those observations — without
executing a single native kernel. Outbound *container* content built via se imports
(`upd_push_tensor@1`) is host-side state whose publication is content-addressed at the publication
record; the guest's decisions about it are fully covered by the asserted calls.

---

## 3. Execution model — the module owns its loop

### 3.1 Inverted control (normative restatement)

The host MUST NOT call the module through an algorithm-shaped lifecycle. On the run instance the
host calls `da_init` once (with the admitted config+grants, §9.4 step 11), then `da_run()` exactly
once; the module runs a long-lived loop that *pulls* events via the blocking `next_event` import and
returns only when it decides to stop or is told to (architecture §3.1). This is the defining
difference from the major-1 driver.

```
da_init(cfg, grants) -> u32   // once; build state, register bridge params (§2.5)
da_run() -> u32               // once; Outcome code (§4.5); blocks inside next_event for the run's life
```

After `da_run` starts, the host invokes **no** guest export to drive the module (non-reentrancy,
§2.4). All host→guest communication flows through the return values of imports the guest calls —
pre-eminently `next_event`.

### 3.2 The two blocking points (normative)

Exactly two guest-visible blocking imports exist at Phase A (architecture §3.3):

- `next_event` — parks the guest thread until the host has an event to deliver (or returns
  `NeedCapacity` without parking when the guest buffer is too small, §4.1).
- `read_back` — parks the guest thread until a requested staged/nondeterministic value is available.

Every other import (Phase A: `publish`, `set_timer`, `cancel_timer`, `now`, `emit_metric`, `log`,
`stage_state`, `snapshot_state`, and the `tabi@1` bridge) MUST return promptly without parking. The async
completion protocol (`OpId` + `Event::Completion`, architecture §3.3) that generalizes non-immediate
calls arrives in Phase B and is reserved (§7.5).

### 3.3 Determinism obligation (normative)

A module MUST be a deterministic function of its **observations**: the ordered sequence of events
returned by `next_event` plus every nondeterministic import result (`read_back` values, clock
readings, publish sequence numbers, timer IDs, bridge `scalar@1` readouts, and — in later phases —
completion results/order, allocation outcomes, device profile). wasm core semantics
(NaN-canonicalized, no ambient I/O) leave no other inputs. Given identical observations, config, and
seeds, a conforming module MUST make identical *decisions* (what it published, to whom, when, what
it claimed) — architecture §3.6 claim 1. The host guarantees this is auditable by journaling exactly
those observations (§8).

The host MUST NOT introduce any guest-observable nondeterminism outside a journaled import result:
no un-journaled timer jitter, no un-journaled queue reordering, no wall-clock leakage, and — per the
generation rule of §7.1 — no host-random handle generations. `now()` returns a host-fed *logical*
clock whose every reading is journaled (§6.5).

---

## 4. The module entry contract and the event vocabulary

### 4.1 `next_event` (normative — guest-provided storage)

Resolution of blocking fix 5 and OQ-3:

```
next_event(buf_ptr: u32, buf_cap: u32) -> u64     // (status << 32) | length
```

- The guest supplies a linear-memory buffer `[buf_ptr, buf_ptr + buf_cap)`. The host validates the
  span (out-of-bounds → `MemOob` trap), serializes the next event as a canonical-CBOR **event
  frame** (§4.2), and:
  - if `frame_len ≤ buf_cap`: writes the frame into the guest buffer and returns
    `(0 << 32) | frame_len` — status **`0 = Delivered`**;
  - if `frame_len > buf_cap`: writes nothing and returns `(1 << 32) | frame_len` — status
    **`1 = NeedCapacity`**, `length` = the exact required capacity. The event is **not consumed**:
    it remains the next pending event. The guest MUST immediately re-call `next_event` with an
    enlarged buffer (`buf_cap ≥ length`); calling any other import first is a `BadEvent` trap
    (the same mandatory-retry rule as `read_back`, §6.4).
- Status values 2–15 are reserved (additive by minor). The guest MUST treat unknown status values
  as fatal and trap (fail closed, §5.2).
- A `NeedCapacity` return does **not** end the current event slice: budgets do not reset (§5.5),
  and no journal record is written for it (the required length is a deterministic function of the
  already-journaled frame, so replay reproduces the exchange).
- A `Delivered` return **starts a new event slice** (§5.5) — except that the very first
  `next_event` call of `da_run` starts the first slice on delivery.
- `next_event` blocks until an event is available (§3.2). It never returns a zero-length frame; the
  end of the run is the explicit `Stop` event (§4.4). Calling `next_event` after consuming `Stop`
  is a `PhaseViolation` trap.
- The host writes only into `[buf_ptr, buf_ptr + frame_len)`; it never calls `da_alloc` (§2.4).

### 4.2 The event set is algorithm-free and closed (normative)

The event set is **mechanism, not vocabulary** (architecture §3.1): there is no `RoundOpen`, no
`Commit`, no algorithm variant. New topologies add *frame schemas* (guest code decoding
`Frame` payloads), never ABI event variants. The complete major-2 event vocabulary, with the Phase A
subset stated closed and later-phase variants reserved:

```cddl
event = frame-ev / payload-ready-ev / timer-ev / budget-ev / stop-ev / quiesce-ev
      / fence-ev / completion-ev              ; RESERVED (Phase C / B) — see §4.6

; --- Phase A closed subset (deliverable at major 2 minor 0) ---
frame-ev         = [0, channel: uint, seq: uint, sender: identity, payload: bstr]
payload-ready-ev = [1, staging_id: uint, hash: bstr .size 32, meta: payload-meta]
timer-ev         = [2, timer_id: uint, fired_at: uint]      ; fired_at = logical ms (§6.5)
budget-ev        = [3, report: budget-report]
stop-ev          = [4, reason: stop-reason]
quiesce-ev       = [7, reason: quiesce-reason, deadline_ms: uint]  ; §4.4

; --- RESERVED (host MUST NOT deliver at minor 0; §4.6) ---
fence-ev         = [5, fence_id: uint]
completion-ev    = [6, op: uint, result: completion-result]  ; completion-result: §7.5

identity     = bstr .size 32     ; ed25519 public key / role-instance identity (proto PeerId)

payload-meta = {
  "size": uint,                  ; staged byte size
  "kind": uint,                  ; staged-kind: 0 = bytes, 1 = batch (bridge, §2.5),
                                 ;              2 = update-container (bridge, §2.5)
  ? "channel": uint,             ; the channel whose frame referenced this payload, if any
}

budget-report = {
  "fuel": uint,                  ; remaining-fuel class: 0 = ample, 1 = low, 2 = critical
  "mem":  uint,                  ; memory-pressure class: 0 = none, 1 = elevated, 2 = critical
  "throttle": {
    "paused":         bool,
    "duty_pct":       uint,      ; 0..=100
    "vram_cap_bytes": uint,      ; 0 = uncapped (raw bytes — §9.6 units rule)
  },
}

stop-reason    = 0 / 1 / 2 / 3   ; 0 = RunComplete, 1 = LeaveRequested, 2 = Fault, 3 = OwnerPolicy
quiesce-reason = 0 / 1 / 2       ; 0 = Upgrade, 1 = Throttle, 2 = reserved (checkpoint barrier)
```

The leading integer is the **event tag** (§5.1). Tags are permanently assigned; a tag is never
reused or renumbered (§5.2). The Phase A deliverable subset is `{Frame, PayloadReady, Timer,
Budget, Stop, Quiesce}` — stated closed: a Phase A host delivers only these six, and a Phase A
module MUST handle all six.

### 4.3 Event semantics (Phase A subset)

- **`Frame`** — a signed control frame the host has received and verified. `channel` is the
  envelope-/table-declared channel it arrived on (§6.2); the channel declaration — not the frame —
  determines the delivery class (*classification is declared, never guest-promoted*, architecture
  §3.1). `seq` is the sender's durable monotonic sequence number **scoped to
  the sender's signed stream `(run_id, epoch, role, instance, channel)`** (§12.2). `payload` is
  opaque module-authored bytes (the
  SDK decodes coordinator records etc.); the host never interprets it.
- **`PayloadReady`** — content-addressed bytes (by blake3) the host has fetched, hash-verified, and
  staged under `staging_id` — the token the guest passes to `read_back` (§6.4). Advisory; deduped
  by hash (§4.7). `meta.kind` distinguishes plain bytes from bridge-staged batches/update
  containers (§2.5).
- **`Timer`** — the one-shot timer `timer_id` (armed with `set_timer`, §6.3) has elapsed on the
  logical clock; `fired_at` is the logical time of delivery. Advisory; queue-depth bounded with
  drop-oldest recorded per §4.7.
- **`Budget`** — a host-initiated budget/pressure/throttle notification with the fully-defined
  `budget-report` body above (resolution of OQ-4). Advisory, latest-wins, with a **host-fixed
  queue depth of 1 for Phase A**: `Budget` is host mechanism, not a subscription — it does not
  appear in `event-caps` and is not negotiable through the manifest. Every delivered `Budget` is
  journaled (the guest may branch on it).
- **`Stop`** / **`Quiesce`** — §4.4.

### 4.4 `Stop` and `Quiesce` (normative — distinct events)

Resolution of blocking fix 6; this supersedes Draft 1's conflation (whose §4.3 "finish using
still-legal imports" contradicted §6.2 "imports after Stop trap").

**`Stop` (tag 4) is terminal.** On decoding `Stop` the guest MUST return from `da_run` promptly
without initiating any new operation: after `Stop` is delivered, **every** import call — including
`next_event`, `publish`, `read_back` — is a `PhaseViolation` trap. The only conforming action is to
compute the Outcome and return. The host delivers no further events after `Stop`.

**`Quiesce` (tag 7) opens a bounded drain.** `Quiesce{reason, deadline_ms}` tells the guest to bring
in-flight work to a consistent point and then return from `da_run` with Outcome `QuiesceReady`.
During the drain:

- **Still-legal imports:** `next_event`, `read_back`, `publish`, `cancel_timer`, `now`,
  `emit_metric`, `log`, `stage_state`, `snapshot_state` (§10.2), and the `tabi@1` bridge ops.
  `set_timer` remains callable but a timer armed during a drain MAY never fire (delivery of new
  `Timer` events is frozen).
- **Still-delivered events:** `Completion` and `Fence` events for already-outstanding operations
  (at minors where they exist — at minor 0 this set is empty), and `Budget`. New `Frame`,
  `PayloadReady`, and `Timer` deliveries are frozen: authoritative frames spool; advisory events
  coalesce per class (architecture §5.4 step 1).
- **Deadline:** `deadline_ms` is expressed on the logical clock. The deadline value is
  **owner-configured, bounded by the lane profile's ceiling** (resolution of OQ-11): the effective
  deadline is `min(owner setting, lane maximum)`, and the host MUST deliver the effective value in
  the event so the guest can plan its drain. On expiry the host forces interruption via the epoch
  mechanism (§11.3); the resulting trap is `QuiesceDeadlineExceeded` (§7.6) and the quiesce is
  recorded as failed (for an upgrade: local rollback per §10.3).
- Drain completion: the guest returns `QuiesceReady` from `da_run`. Returning `Ok`/`Left` from a
  drain is permitted and means the module chose to leave rather than quiesce (journaled verbatim).

Upgrade and throttle both use `Quiesce` first and forced interruption only on expiry (§10.3, §11.3).

### 4.5 `da_run` return (Outcome)

`da_run() -> u32` returns an Outcome code the host records and surfaces to the node:

| Code | Name | Meaning |
|---|---|---|
| 0 | `Ok` | Clean finish after a `Stop`. |
| 1 | `Left` | The module chose to leave the run (its own policy). |
| 2 | `QuiesceReady` | Returned during a `Quiesce` drain; snapshot manifest published (§10.2). |
| 3 | `StaleRestore` | The module refuses to fold a record history **gapped above its restored resync watermark** (rounds committed before this incarnation attached are never re-delivered on the ordered records channel; folding across them would fork the det trajectory). The node treats this outcome as **retryable**: the recovery is a rejoin restoring a fresher checkpoint, and live pointers advance every ingested round, so the retry converges on the live edge. |
| 4–15 | reserved | assigned only by a future minor of this document. |
| ≥16 | module-defined | journaled verbatim; treated by the host exactly as `Left`. |

(Resolution of OQ-5: the vocabulary stays small; module-specific exit information belongs in a
published frame or the snapshot manifest, not in the Outcome. Unknown reserved codes 4–15 from a
future-minor module never reach an older host by §1.4; a host receiving one anyway treats it as
`Left` and journals it — which is also the sound degraded reading of `StaleRestore` on a host
older than its assignment.)

A guest that returns from `da_run` without having consumed a `Stop`/`Quiesce` (it fell out of its
own loop) is treated as `Left` with a journaled warning. A trap during `da_run` is handled per §7.6
and is *not* an Outcome — it is a typed local error; the subprocess survives and re-instantiation is
a normal recovery path (architecture §3.5).

### 4.6 Reserved variants (Fence / Completion)

`Fence` (tag 5) and `Completion` (tag 6) are **reserved** and MUST NOT be delivered at major 2
minor 0:

- **`Fence(fence_id)`** — a compute-queue marker the guest inserted with `compute.fence(id)` has
  been passed by the device (architecture §3.3). Arrives with `compute@2` in **Phase C**.
- **`Completion(op, result)`** — the generalized async result for any capability call that could
  not complete immediately (architecture §3.3). Arrives with the completion protocol in **Phase
  B**. The `completion-result` encoding is fixed now in §7.5 so journal and SDK shapes are stable.

A module built at a higher minor that legitimately receives these decodes them by tag; a minor-0
module never sees them (§1.4) — and if it ever did, it MUST fail closed (§5.2).

### 4.7 Delivery ordering guarantees (normative)

Two delivery classes, with the classification **declared per channel** (§6.2), never guest-selected:

- **Authoritative channels**: reliable and **ordered per sender per channel**. The host MUST
  deliver authoritative frames from a given `(sender, channel)` in strictly increasing `seq` order,
  with:
  - dedup by the full signed-stream tuple `(run_id, epoch, role, instance, channel, seq)`
    (§12.2) within the channel's declared
    `replay_window`;
  - explicit **sequence-gap detection** on the full scope tuple (§12): on a detected gap the host
    backfills from the record archive (Phase D) or, pre-archive, back-pressures the network
    reader; it MUST NOT silently skip. An unrecoverable gap raises run condition
    `SequenceGapUnrecoverable` (§6.7).
  - a bounded durable spool with the channel-declared `max_frame_bytes`, `spool_frames`, and
    `per_sender_quota`; genuine spool exhaustion raises run condition `SpoolExhausted` (§6.7) —
    never a silent drop. A malicious sender MUST NOT be able to use the reliable class as an
    unbounded memory-DoS vector.
- **Advisory events** (`PayloadReady`, `Timer`, gossip-class channels — plus `Budget`, whose
  depth is host-fixed at 1 and never manifest-negotiated, §4.3): bounded per-class queues with
  **manifest-declared depth** (§2.3) and a **fixed coalescing rule per class**:
  - `PayloadReady` → dedup by hash (coalesce code 0);
  - `Timer`/`Budget` → latest-wins (code 1) — for timers, "latest-wins" operates on the queue,
    dropping the oldest queued `Timer` events beyond the declared depth (each one-shot `timer_id`
    still fires at most once, §6.3);
  - gossip → drop-oldest (code 2).
  Every drop or coalesce MUST be journaled (§8.3 tag 7), so replay is exact regardless of what was
  dropped. Because advisory channels/classes have **no sequence-gap semantics** and authoritative
  gap detection operates on the per-channel scope tuple, an advisory drop can never manufacture or
  mask an authoritative gap (resolution of blocking fix 9's corollary).

Cross-class ordering (an advisory event relative to an authoritative one) is **not** guaranteed and
MUST NOT be relied on. Resolution of OQ-6: authoritative-channel bounds (`replay_window`,
`spool_frames`, `per_sender_quota`, `max_frame_bytes`, `rate_per_min`) are **declared per channel**
— in the Phase-A default channel table now, in the genesis envelope from D0 (§6.2) — and are
**constrained by the lane profile's ceilings** at admission: a channel table exceeding the lane's
ceilings is refused `GrantsExceedLane`.

---

## 5. Event frame encoding, versioning, budgets, and the watchdog

### 5.1 Frame encoding (normative)

An event frame is canonical CBOR: a definite-length array whose **first element is the integer
event tag** (§4.2), followed by tag-specific fields. Rationale for a positional array over a map:
the tag is the dispatch key, frames are hot, and positional arrays are the smallest canonical form.
Identities, hashes, and payloads are byte strings (`bstr`); sequence numbers, IDs, and enum values
are unsigned integers with the assignments fixed in §4.2.

Frames are the wire the journal stores verbatim (§8.3 tag 1), so their canonical encoding is the
audit substrate.

### 5.2 Additive versioning rules — unknown input fails closed (normative)

- **Tags are permanent.** An event tag, once assigned, denotes exactly one variant forever. New
  variants take new tags at a higher minor. Tags are never renumbered, reused, or removed.
- **Fields are append-only within a tag.** A new minor MAY append trailing OPTIONAL fields to an
  existing tag's array. A decoder MUST accept and ignore trailing fields beyond those it knows
  (definite-length arrays make this unambiguous). A new minor MUST NOT change the type or meaning
  of an existing positional field or enum value.
- **Unknown tags fail closed** (supersedes Draft 1's advisory no-op). The **admission-time**
  guarantee is minor negotiation: the host MUST NOT deliver a variant, trailing field, or enum
  value introduced at a minor above the module's declared minor (§1.4) — so a conforming pairing
  never produces an unknown tag. If a module nonetheless decodes an event whose tag it does not
  know, it MUST **trap immediately** (the SDK executes `unreachable`, surfacing as `GuestPanic`
  with the tag in the detail) — it MUST NOT skip, guess, or continue. Silently ignoring an unknown
  input on a consensus-bearing surface is a divergence vector; a trap is a contained, attributable,
  journaled local error.
- The same fail-closed rule applies to unknown `next_event` status values (§4.1), unknown
  `read_back` status values (§6.4), and unknown enum values in any host-authored structure.

### 5.3 Determinism of encoding

For a given logical event and a given negotiated tuple (§1.4), the host MUST produce a
byte-identical frame every time (canonical CBOR + fixed field order). Two hosts at the same tuple
replaying the same journal MUST produce identical frames. This is required for §8 replay and for
any digest taken over the delivered stream.

### 5.4 Delivery-class enforcement

At admission the host validates `manifest.events` advisory depths/coalescing and the channel table's
authoritative bounds against the lane's ceilings (§9.6). At runtime the host enforces exactly the
declared depths and the fixed coalescing rule per class (§4.7). Because these are declared and
checked, event-delivery behavior can never silently become a host-side scheduler; and because the
journal records the *delivered* sequence, an implementation change to the scheduler cannot change
replay.

### 5.5 Budget semantics per event slice (normative)

An **event slice** is the guest computation between one `Delivered` return of `next_event` and the
next `next_event` call that yields `Delivered` (or the `da_run` return). Budgets are **per-slice**
(architecture §3.5; refactor §5 A2):

- **Fuel** — wasmtime fuel meters guest instructions. The host MUST reset the per-slice fuel
  allowance at each `Delivered` return (a `NeedCapacity` return does not reset, §4.1). Exhaustion
  traps `BudgetFuel`.
- **Op budget** — the count of capability import calls in a slice — **including every `tabi@1`
  bridge call** (§2.5) — is capped (`op_budget` in `EngineConfig` today). Reset per slice.
  Exhaustion traps `BudgetOps`.
- **Live resources** — the live instance-class resource count and byte total are capped
  continuously against `buffer-req` (§2.3) — standing quotas, not per-slice. Exhaustion traps
  `BudgetHandles`.
- **Linear memory** — the linear-memory cap is a standing limit (wasmtime `StoreLimits`).
  Exhaustion traps `BudgetMemory`.
- **Readback bytes** — bytes written into linear memory by `read_back` are capped per slice by
  `buffer-req.max_readback_bytes`. Exhaustion traps `GrantViolation` naming the grant.

`read_back` executes **within** the current slice and is charged against that slice's fuel/op/byte
budgets; the blocked wait itself consumes no fuel (the guest thread is parked). This in-slice
charging is normative (resolution of OQ-7; a long readback that needs its own window is a Phase C
`compute@2` concern, to be ratified there).

Budget accounting MUST be deterministic and journaled where the guest can observe it (a `Budget`
event is journaled; a trap is a terminal journaled fact). Fuel/op *limits themselves* are host/owner
policy, not guest-observable except via `Budget` events and traps — so two hosts with different
limits are not required to produce identical journals, but a *single* journal replays identically
(§8.7).

### 5.6 Epoch watchdog interaction (normative)

The epoch deadline (wasmtime epoch interruption, the `EpochThread` ticking `increment_epoch`) is the
**watchdog against a guest that never returns to `next_event`** — a pure-compute spin:

- The epoch deadline arms when a slice starts (a `Delivered` return hands control to the guest) and
  disarms while the guest is **parked inside** `next_event` or `read_back` (a parked guest is not
  spinning; it must not be killed for waiting on the host).
- If the guest burns wall-clock past the deadline **within a slice** (not parked), the epoch fires
  and the slice traps `BudgetEpoch` — the v1 per-call watchdog, scoped to the slice. The same
  mechanism enforces the quiesce deadline (`QuiesceDeadlineExceeded`, §4.4) and forced throttle
  teardown (§11.3); epoch interruption is the **only** sanctioned way to interrupt a running guest,
  because it is cross-thread-safe and surfaces the trap **on the guest thread** (§11.3).
- The epoch watchdog is orthogonal to fuel: fuel bounds *instructions* (deterministic), epoch
  bounds *wall-clock* (non-deterministic). **Replay rule (resolution of OQ-8, normative):** an
  epoch-class trap is recorded as a terminal fault at a recorded journal ordinal (§8.3 tag 9);
  replay MUST NOT re-arm any wall-clock mechanism — it re-drives the guest from recorded
  observations up to that ordinal and then **injects the recorded terminal fault** as the replay
  outcome. Replay never reproduces wall-clock behavior; it reproduces the recorded consequence.

---

## 6. Capability imports — the Phase A closed subset

Phase A links the following imports and no others (refactor §5 A2: "Phase A's exact minimal
capability subset, stated closed") — plus the transitional `tabi@1` bridge (§2.5) and the
quiesce-scoped `snapshot_state` (§10.2). Everything else (`compute@2`, `data@2`, gossip/streams,
payload put/get, the completion protocol) is reserved and arrives in B/C.

### 6.1 The subset (normative, closed)

| Import | Signature | Blocking | Journaled |
|---|---|---|---|
| `vhc@2::next_event` | `(buf_ptr: u32, buf_cap: u32) -> u64` | yes | delivered event (§8.3 tag 1) |
| `vhc@2::read_back` | `(src: u64, kind: u32, out_ptr: u32, out_cap: u32) -> u64` | yes | value/status (§8.3 tag 2) |
| `vhc@2::stage_state` | `(ptr: u32, len: u32) -> u64 /*staging_id*/` (§10.2) | no | no (deterministic; guest bytes) |
| `vhc@2::snapshot_state` | `(manifest_ptr: u32, manifest_len: u32) -> u32` — quiesce-only (§10.2) | no | manifest (§8.3 tag 10) |
| `net@2::publish` | `(channel_id: u32, payload_ptr: u32, payload_len: u32) -> u64 /*seq*/` | no* | outbound frame (§8.3 tag 4) |
| `sys@2::set_timer` | `(delay_ms: u64) -> u64 /*timer_id*/` | no | arm (§8.3 tag 5) |
| `sys@2::cancel_timer` | `(timer_id: u64) -> u32 /*status*/` | no | cancel outcome (§8.3 tag 6) |
| `sys@2::now` | `() -> u64 /*logical ms*/` | no | clock reading (§8.3 tag 3) |
| `sys@2::emit_metric` | `(name_ptr: u32, name_len: u32, value: f64)` | no | no (egress only) |
| `sys@2::log` | `(level: u32, msg_ptr: u32, msg_len: u32)` | no | no (rate-limited egress) |
| `tabi@1::*` (bridge) | frozen v1 signatures (§2.5) | no | kinds ≥ 128 where nondeterministic (§2.5) |

\* `publish` does not park the guest on the network, but it MUST NOT return before its durability
barrier commits (§6.2, §8.4) — a bounded local-disk wait, not an unbounded network wait.

### 6.2 `publish` and the channel table (normative)

Resolution of blocking fix 10 and OQ-9/OQ-15. The guest **selects a channel ID**; it never supplies
a class, a topic, or any routing information:

```
publish(channel_id: u32, payload_ptr: u32, payload_len: u32) -> u64   // the stamped seq
```

- The host resolves `channel_id` against the **channel table**. The channel declaration — not the
  guest — determines delivery class, routing/topic, size and rate bounds:

```cddl
channel-decl = {
  "id":               uint,
  "name":             tstr,
  "class":            uint,   ; 0 = authoritative, 1 = advisory/gossip
  "direction":        uint,   ; 0 = rx-only, 1 = tx-only, 2 = bidirectional
  "max_frame_bytes":  uint,
  "rate_per_min":     uint,   ; tx token bucket
  ; authoritative channels only:
  ? "spool_frames":     uint, ; bounded durable spool depth
  ? "replay_window":    uint, ; dedup window (frames)
  ? "per_sender_quota": uint, ; rx per-sender outstanding quota
}
```

- **Phase A default channel table** (envelope v2 does not exist until D0): the table is a
  **driver-provided constant in `daemon-vhc-abi`**, versioned with the ABI minor. At minor 0 it
  declares exactly one channel:

| id | name | class | direction | notes |
|---|---|---|---|---|
| 0 | `control` | authoritative (0) | bidirectional (2) | maps onto today's `SignedMessage` control plane; bounds from the lane profile |

  Publishing on any undeclared or rx-only channel traps `GrantViolation` (detail names the
  channel). A module's manifest MUST list every channel it uses (§2.3); a manifest channel absent
  from the table is refused `GrantsExceedLane` at admission. **From D0, channel declarations move
  into the genesis envelope** (per-role channel tables in the role's grant list); the ABI surface —
  `publish(channel_id, …)`, `frame-ev.channel`, the scope tuple of §12 — is unchanged by that move.
- The host wraps the payload in the domain-separated signed envelope (§12), stamps the durable
  monotonic `seq` scoped to its own signed stream `(run_id, epoch, role, instance, channel)`
  (§12.2), signs with the per-run key, and
  returns the seq. **The return commits atomically** (resolution of OQ-9): `publish` MUST NOT
  return until (a) the seq is allocated from the durable counter, (b) the outbound journal record
  (§8.3 tag 4, containing the full signed frame) is written, and (c) the frame is inserted into the
  durable spool — all covered by one durability barrier (§8.4). Network transmission happens
  afterwards, asynchronously; transmission outcomes are not guest-visible at Phase A (they become
  `Completion` events for the operations that need them in Phase B).
- **Sequence semantics are final from Phase A** (resolution of OQ-15): the frame is signed under
  the §12.1 domain-separated envelope **from A2 onward**, and the returned seq is the durable,
  channel-scoped sequence number of §12.2 with its full evidentiary meaning. There is no interim
  counter, no interim frame schema, and no later reinterpretation: D1 adds certified key chains
  and `Authority` around the frozen envelope, never fields to it (§12.1).
- `payload_len` exceeding the channel's `max_frame_bytes` traps `PayloadOverflow`; exceeding the
  rate bound traps `GrantViolation`.

### 6.3 Timers (normative)

- `set_timer(delay_ms)` arms a **one-shot** timer on the logical clock and returns a fresh
  `timer_id`. Timer IDs are **plain typed u64 values scoped to one guest instance** — not resource
  handles, never subject to the §7 handle layout (resolution of OQ-10). They are assigned by a
  deterministic per-instance counter starting at 1 and incrementing by 1 per arm; IDs are **never
  reused within an instance** (the counter is monotone) and are meaningless across instances.
  Because the counter is deterministic, timer IDs are replay-reproducible; each arm is nonetheless
  journaled (§8.3 tag 5) with its logical arm time.
- **Expiry**: the timer fires at the first event-pump dispatch where
  `logical_now ≥ armed_at + delay_ms`. Delivery is a `Timer` event carrying `fired_at`. Each
  `timer_id` fires **at most once**; duplicate delivery is forbidden.
- **Rearming**: there is no rearm operation. Rearming = `cancel_timer(old)` + `set_timer(new)`,
  yielding a new ID.
- **Cancellation**: `cancel_timer(timer_id) -> u32` returns `0 = Cancelled` (the timer had not
  fired and will never be delivered — the host MUST NOT deliver its `Timer` event after a `0`
  return, including one already queued) or `1 = AlreadyFiredOrUnknown` (it fired, was already
  delivered, was already cancelled, or the ID was never issued). Statuses 2–15 reserved; unknown
  statuses fail closed (§5.2). The outcome is journaled (§8.3 tag 6) — it is a nondeterministic
  observation (whether the cancel raced the firing).
- A timer armed during a `Quiesce` drain MAY never fire (§4.4).

### 6.4 `read_back` (normative)

The explicit, budgeted, journaled blocking readback (architecture §3.2/§3.6; refactor §5 A2):

```
read_back(src: u64, kind: u32, out_ptr: u32, out_cap: u32) -> u64   // (status << 32) | length
```

- **`src` is a staging ID, not a handle** (resolution of the review's src question): the 64-bit
  `staging_id` announced in a `PayloadReady` event (§4.2), carried in a migration descriptor
  (§10.2), or returned by `stage_state` (§10.2). Staging IDs form their own namespace — they are
  never valid where a §7 handle is expected and vice versa; they do not use the handle bit-layout.
  Host-assigned IDs (top bit clear) are unique per instance and reach the guest only through
  delivered events or the migration descriptor; guest-created IDs (`stage_state`, top bit set) are
  counter-deterministic — so replay is closed either way.
- **`kind` enumeration** (assigned; additive by minor; ≥ 128 reserved for bridge-op journal kinds,
  §2.5, which are never valid as call arguments):

| kind | name | returned bytes | notes |
|---|---|---|---|
| 0 | `staged-bytes` | the staged payload bytes (raw) | any `PayloadReady` with `meta.kind = 0` |
| 1 | `staged-batch` | canonical CBOR `uint` — a `tabi@1` batch handle | bridge only (§2.5); `meta.kind = 1` |
| 2 | `staged-update` | canonical CBOR `uint` — the staging index for `upd_*@1` | bridge only (§2.5); `meta.kind = 2` |
| 3 | `state-section` | bytes of a named state-manifest section | migration restore (§10.2); requires the restore grant; **legal during `da_migrate`** (§6.6) |
| 4–127 | reserved | — | assigned only by a future minor |

- **Return packing** (same scheme as `next_event`): `(status << 32) | length` with
  `0 = Ok` (`length` bytes written to `[out_ptr, out_ptr+length)`, `length ≤ out_cap`) and
  `1 = NeedCapacity` (nothing written; `length` = required capacity; the staged value remains
  available). Statuses 2–15 reserved; unknown fails closed (§5.2). A `src`/`kind` pair that names
  nothing stageable traps `ReadBackUnavailable`; a kind the grants don't allow traps
  `GrantViolation`; an out-of-bounds span traps `MemOob`.
- **`NeedCapacity` is a mandatory-retry protocol rule** (adopted): on `NeedCapacity` the guest
  MUST immediately re-call `read_back` with the same `(src, kind)` and an enlarged buffer
  (`out_cap ≥ length`); calling any other import, calling `read_back` with different arguments, or
  returning from the current context before the retry is a `BadEvent` trap. Because the retry is
  forced and the required length is a deterministic function of the journaled value, the
  `NeedCapacity` status needs **no journal record** — replay reconstructs the exchange from the
  recorded `Ok` value alone (rationale recorded per ratification).
- **Every `Ok` return is journaled** (§8.3 tag 2) with the exact bytes delivered (inline or via
  sidecar, §8.5), so replay re-feeds the value bit-exactly without re-executing whatever produced
  it.
- Charged in-slice per §5.5.

### 6.5 `now`, logical time, `emit_metric`, `log` (normative)

> **Extended at the certification minor.** §17.6 adds the `da_init`/`da_migrate` exemption for `log`
> and its ordered bounds; that section governs where the two differ.

- **Logical time** (adopted definition) — a `u64` millisecond value with:
  - **Epoch (zero point): run join.** Logical time 0 is the creation of the run-instance journal
    (the tag-0 run header). Chosen over process start because the timeline must be a property of
    the *run instance*, not of whichever OS process happens to host it across restarts.
  - **Sampling.** The event pump samples the **host monotonic clock once per delivered event, at
    delivery time, before delivery**, converts it to logical time (monotonic delta since journal
    open + the journal's logical base), and journals it with the event (§8.3 tag-1 `at` field).
    No other sampling point exists.
  - **Slice-constant `now()`.** `now()` returns the logical timestamp of the current slice's
    delivered event — constant within a slice. During `da_init` (before the first delivered
    event) it returns the instantiation's logical time (§8.3 tag-13 `at` field). Every reading is
    nonetheless journaled (§8.3 tag 3) — the v1 coordinator-replay lesson: clocks are not
    messages but must be captured.
  - **Monotonicity across restarts.** On journal open (initial or after crash/restart) the
    logical base is clamped to the journal's **high-water mark** (the maximum logical time in any
    committed record), so logical time is monotone non-decreasing across the whole journal;
    the first post-restart sample MUST be ≥ the high-water mark.
  - **Replay** uses recorded readings only (tag-1 `at`, tag-3, tag-13 `at`); no clock is sampled
    at replay. Simulation virtualizes time by feeding the pump a virtual monotonic clock
    (architecture §3.2).
- **`emit_metric`** — egress-only telemetry. Rules (adopted correctness fix):
  - `value` MUST be finite. A NaN or ±Inf value causes the metric to be **dropped host-side**
    (counted against the rate limit, optionally logged by the host) — never a trap, never a
    journal record, never forwarded. Metrics are advisory egress; a numeric pathology must not be
    a crash vector.
  - Rate limiting is a host-configured token bucket per instance (the grant's `rate_per_min`
    bounds it, §2.3); over-limit calls are silently dropped. `name_len` > 128 bytes → dropped.
  - The import returns nothing; the guest MUST NOT infer acceptance. Not journaled (an output, not
    an input) but bounded by grants — metrics are an egress channel (architecture §10).
- **`log`** — rate-limited egress with the same drop semantics; `level` values above the host's
  maximum are clamped. Not journaled.

### 6.6 Legality (no phase table for v2)

The v1 phase-legality table (`phase.rs`) is **retired for v2** (refactor §5 A2). In its place, three
temporal rules enforced by the dispatch layer:

1. **Before `da_run`**: no capability import is legal. In the assessment instance every capability
   import is a deny-on-call stub trapping `ClaimCapabilityDenied` (§9.2). In the run instance,
   imports called during `da_init` trap `PhaseViolation` — with the single exception of the
   `tabi@1` registration imports, which are legal **only** there (§2.5). During **`da_migrate`**
   exactly one import is legal: `read_back` with `kind = 3` (state-section restore, §10.2); any
   other import — or `read_back` with any other kind — traps `PhaseViolation`.
2. **During `da_run`**: all §6.1 imports are legal in any slice, except `snapshot_state` (legal
   only during a `Quiesce` drain, §10.2) and subject to the drain restrictions of §4.4
   (`stage_state` is legal in any slice and during a drain).
3. **After `Stop` is consumed**: no import is legal (`PhaseViolation`), §4.4.

The det-lane/consensus discipline the v1 phase table encoded moves to **SDK types**
(`Committed<T>`, `DetTensor` constructible only from committed inputs — architecture §4.2), enforced
at compile time for cooperative authors and bounded by sandbox/grants for adversarial ones.

### 6.7 Run-condition codes (normative)

Typed, journaled, node-visible conditions that are neither guest traps nor admission refusals
(completing the wire-gap list: spool exhaustion, sequence gaps, archive availability):

| Code | Meaning | Host behavior |
|---|---|---|
| `SpoolExhausted` | an authoritative channel's durable spool hit `spool_frames` | back-pressure the network reader; journal condition (§8.3 tag 16); notify node; if sustained past owner-configured tolerance → deliver `Stop{Fault}` and leave the run |
| `SequenceGapUnrecoverable` | a gap in a signed stream `(run_id, epoch, role, instance, channel, seq)` (§12.2) could not be backfilled (no archive / archive lacks the range) | journal condition; deliver `Stop{Fault}`; surface typed error to node |
| `ArchiveUnavailable` | record-archive fetch required (catch-up/backfill, Phase D) and unavailable | journal condition; retry per host policy; escalate to `Stop{Fault}` on exhaustion |

Each condition record carries `{code, detail}`; the code enumeration is additive by minor.

---

## 7. Resources, handles, and the trap/error taxonomy

### 7.1 Three resource classes (normative)

Resolution of blocking fix 8. Every host-side resource a guest can name belongs to exactly one
class, which fixes its lifetime, its handle generation behavior, and its restart semantics:

| Class | Members (Phase A / bridge; later phases) | Lifetime | After instance restart |
|---|---|---|---|
| **Registered** | bridge params / persistents / det-persistents (registered in `da_init`); future registered resources where determinism justifies it | the run instance | **Re-derived**: handles are a deterministic function of 1-based registration order within their kind (generation 0), so the same `da_init` yields the same handles |
| **Instance** | bridge staged batches + update containers (§2.5); buffers (kind 8) and streams (kind 9) in Phase B; OpIds (kind 10) in Phase B | until explicit release (`drop@1` / `release`) or instance end | **Invalid**: generational handles from a dead instance MUST trap `StaleHandle`; the guest re-acquires (re-stage, re-open) through capability calls |
| **Slice** | bridge step tensors (kinds 1–2); future per-slice scratch | until the current event slice ends | invalid (a fortiori) |

**Generation determinism (normative nuance, honored):** handle generations MUST be
replay-deterministic — they derive from **journaled instantiation counters, never host randomness**.
Concretely:

- Each (re-)instantiation of a module for a run increments a per-run **instantiation counter**,
  journaled as an `instantiation` record (§8.3 tag 13) before any guest code runs.
- Instance-class arena generations are seeded from that counter and advance deterministically on
  slot reuse (increment-by-one, exactly the v1 `StepArena` discipline in `handle.rs`).
- Slice-class arenas bump generations wholesale at each slice boundary, deterministically.
- Registered-class handles carry generation 0 always.

Replay of a journal therefore reproduces every handle value bit-exactly.

**Generation wrap / ABA (normative):** generations are 24 bits. A slot whose generation would wrap
past `0xFF_FFFF` MUST be **permanently retired** — never returned to the free list; the arena
allocates a fresh index instead. This makes ABA reuse of a `(kind, generation, index)` triple
impossible within an instance. Index exhaustion (2³²−1 indices per kind) is a `BudgetHandles`
situation long before it is reachable in practice.

### 7.2 Handle encoding and kinds

The v1 bit layout (`handle.rs`) is retained. A handle is an opaque nonzero `u64`; `0` is never a
live handle:

```
handle = (kind << 56) | ((generation & 0xFF_FFFF) << 32) | index      // index: 32 bits, 1-based
```

| kind | class (§7.1) | resource | phase |
|---|---|---|---|
| 1 | slice | step tensor (native) — bridge | A (bridge) |
| 2 | slice | step tensor (det) — bridge | A (bridge) |
| 3 | registered | param — bridge | A (bridge) |
| 4 | registered | persistent — bridge | A (bridge) |
| 5 | registered | det persistent — bridge | A (bridge) |
| 6 | instance | update container — bridge | A (bridge) |
| 7 | instance | batch — bridge | A (bridge) |
| 8 | instance | **BufferHandle** (§7.4) | B (reserved) |
| 9 | instance | **StreamHandle** | B (reserved) |
| 10 | instance | **OpId** (§7.5) | B (reserved) |
| 11–255 | — | reserved; assigned only by a future minor | — |

Timer IDs and staging IDs are **not handles** (§6.3, §6.4) and never use this layout.

### 7.3 Lifetime obligations

- The host MUST invalidate all slice-class handles at each slice boundary (wholesale generation
  bump), so any retained slice handle traps `StaleHandle` in a later slice.
- On trap or re-instantiation the host MUST force-reclaim every resource via the per-instance
  handle table (architecture §3.4); the new instance's journaled instantiation counter (§7.1)
  guarantees every stale handle from the dead instance decodes to a wrong generation.
- Instance-class resources are explicitly released by the guest and counted against
  `buffer-req.max_live_handles`/`max_live_bytes`; exceeding a quota traps `BudgetHandles`.

### 7.4 Buffers (reserved Phase B, shape fixed)

`BufferHandle` (kind 8) is the opaque, host-owned, **sealed** byte region every world speaks
(architecture §3.4). Reserved for Phase B; the handle kind, resource class (instance), trap codes,
and the two budgeted linear-memory crossing paths (`read_into`, `create_from`) are fixed now so
Phase B lands without renumbering. A buffer from `payload_get`/`data.fetch` is hash-verified
**before** its completion is delivered.

### 7.5 Reserved: async completion protocol (Phase B) — result encoding fixed now

Any capability call that cannot complete immediately returns an `OpId` (kind 10) and completes via
`Completion(op, result)` (tag 6). `vhc@2::cancel(op)` completes with `Cancelled`. Direct streams use
credit-based flow control. None of this is linked at Phase A, but the wire encoding is fixed now
(wire-gap resolution) so journals and SDKs are stable:

```cddl
completion-result = [0, success-payload]      ; success
                  / [1, comp-error]           ; failure

success-payload = uint                        ; a handle (buffer/tensor/stream) — kind-tagged per §7.2
                / bstr .size 32               ; a content hash (payload_put)
                / null                        ; unit success (stream_write, publish-ack, cancel-target)

comp-error = {
  "code":   uint,      ; 0 = Cancelled, 1 = NetUnreachable, 2 = Timeout, 3 = StoreRefused,
                       ; 4 = HashMismatch, 5 = CreditExhausted, 6 = PeerClosed,
                       ; 7 = GrantExhausted; 8–63 reserved (additive by minor)
  ? "detail": tstr,
}
```

The result-variant set is additive within major 2 (§1.4); unknown codes fail closed (§5.2).

### 7.6 Trap and error taxonomy (normative)

> **The context enumeration below is superseded at the certification minor** by the eleven-value,
> minor-selected domain of §17.5. The code taxonomy in this section is unchanged and remains normative.

Host imports never return status codes for programming errors — they **trap** immediately with a
typed code (v1 T4 discipline, `trap.rs`). wasmtime's own traps (fuel/epoch/memory/`unreachable`/oob)
are mapped into the same taxonomy so a trapping module is a typed local error, never a worker crash.
A trap carries `{code, import, context, detail}` where `context` is the slice ordinal or one of
`da_init` / `da_run` / `da_claim` / `da_manifest` / `da_migrate` / `assessment`.

Retained v1 codes (verbatim slugs, `trap.rs`): `InvalidHandle`, `StaleHandle`, `LaneMismatch`,
`PhaseViolation`, `ShapeMismatch`, `DtypeMismatch`, `RankOverflow`, `MemOob`, `AllocFail`,
`PayloadOverflow`, `BudgetFuel`, `BudgetEpoch`, `BudgetMemory`, `BudgetHandles`, `BudgetOps`,
`GuestPanic`, `NameCollision`, `NotScalar`, `BadEnum`, `AbiMismatch`, `BadModule`. (`AbiMismatch`
and `BadModule` additionally serve as the internal umbrella / refusal codes of §1.5; as *traps* they
are v1-driver-only.)

Semantic scope for v2:

- `PhaseViolation` — an import called outside its temporal window (§6.6): during `da_init` (except
  bridge registration), after `Stop`, `snapshot_state` outside a drain, `next_event` after `Stop`.
- `LaneMismatch`, `ShapeMismatch`, `DtypeMismatch`, `RankOverflow`, `NotScalar`, `NameCollision` —
  live under the `tabi@1` bridge (§2.5) with their v1 meanings; carried forward by `compute@2`.
- `PayloadOverflow` — a `publish` payload exceeding the channel's `max_frame_bytes` (§6.2).

New v2 trap codes:

| Code (slug) | Meaning |
|---|---|
| `ClaimCapabilityDenied` | a capability import was called in the assessment instance (deny-on-call stub, §9.2) |
| `GrantViolation` | a call exceeded a grant/channel bound (undeclared channel, rate, readback bytes, kind not granted) |
| `BadEvent` | the guest violated the event protocol in a way the host detects (e.g. re-entering `next_event` with an invalid span pattern after `Stop` is covered by `PhaseViolation`; malformed guest-supplied CBOR where the ABI requires it) |
| `ReadBackUnavailable` | `read_back` named a `(src, kind)` that stages nothing |
| `MigrateBudget` | `da_migrate` exceeded its bounded fuel/memory (§10.3) |
| `QuiesceDeadlineExceeded` | the guest failed to return from a `Quiesce` drain before the effective deadline; forced epoch interruption (§4.4, §11.3) |

`MigrateUnsupported` is **not** in this table: it is an admission refusal (§1.5), never a trap.
Every slug MUST be unique and stable (the v1 uniqueness test extends). Traps during `da_run` are
local errors: the subprocess survives, the instance is torn down **by the guest thread** (§11.3),
and (per owner/coordinator policy) the module leaves the run or trap-and-restarts (µs
re-instantiation from cached `InstancePre`, with a fresh journaled instantiation counter, §7.1).

---

## 8. The journal

The journal is the host capability that makes policy-determinism (architecture §3.6 claim 1)
operational: it records **everything the guest could have branched on** so that re-feeding it
reproduces every decision bit-for-bit. It generalizes today's `daemon-swarm-observe` capture (the
in-memory `Vec<Input>` `MessageLog` + `RunCapture`) into a **crash-safe, segmented, append-only**
on-disk journal (refactor §5 A1), built and audited **before** the event loop ships.

### 8.1 Execution identity and what is recorded

**The execution identity is the frozen five-tuple**

```
(run_id, epoch, role, instance, module_hash)
```

where `run_id` is the genesis/frozen-envelope hash, `epoch` the transition-chain position,
`role` the envelope-level role label, **`instance` the node-assigned role-instance incarnation id**
— a **never-reused, node-durable, monotonic `u64`** disambiguating N concurrent sandboxes of one
role on one host (decisions D1; **erratum, §0.5**: this is deliberately *not* a reusable small
ordinal/`u16` slot — a reusable slot value would let a fresh incarnation inherit or collide with a
retired incarnation's durable sequence stream (§12.2), so the value that enters the execution
identity, journal, sidecar ownership, and sequence scopes is the never-reused incarnation, while a
reusable local supervision slot, if any, is a node-side concern that never appears here). The
incarnation is **stable across trap-restarts within a join** — distinct from the tag-13
*instantiation counter*, which counts restarts *within* one incarnation — and `module_hash` is the
pinned blob. This tuple keys the
journal and appears in every segment header (§8.2), the run-header record (§8.3), every sidecar
header (§8.5), the signed-frame envelope and domain separation (§12), and the admission machine's
run-header journaling (§9.4 step 10). Sequence scopes derive from it per §12. The **run-key
certificate** binds a role-instance's per-run signing key to this exact five-tuple, so a signature
is authenticated only for the precise `(run, epoch, role, incarnation, module)` it was certified
for (§12.3); the same five-tuple is the artifact-addressed core of the **admitted tuple** admission
produces and join re-verifies (architecture §6.3.2).

Per execution identity, the journal records, in observation order: the run header (admitted
manifest/config/grants/claim/device profile); every instantiation; the `da_init` call; every
delivered event frame verbatim with its logical delivery time; the original signed wire frame (or
evidence reference) behind every authoritative `Frame` event; every nondeterministic import result
(`read_back` values, clock readings, publish seqs, timer arms and cancel outcomes, the bridge
registry's nr-class results — §2.7); every advisory drop/coalesce; every throttle change; every run
condition (§6.7); and the terminal fact (Outcome, trap, or forced interruption). Bulk payloads are
**not** copied — they are content-addressed and re-fetched at replay (§8.7); large readback values
go to sidecars (§8.5).

**Continuation across a live upgrade (normative).** A live module upgrade (§10.3) advances the
epoch and mints a new never-reused role-instance incarnation, so the post-upgrade instance runs
under a **new execution identity** — but the host does **not** start a new journal. The log is
*continued in one file series*: the retired incarnation's records remain as the prefix and, at the
upgrade seam, the incoming incarnation **rolls the current segment and opens the new one with its
own run-header record** (tag 0, §8.3) carrying its full new execution identity
`(run_id, epoch, role, instance, module_hash)`, after which its records append under that identity.
One journal therefore holds a run-header-delimited sequence of execution-identity spans across the
run's upgrades; the per-journal record ordinal (`ord`) remains globally monotone across every seam,
and — because the seam forces a segment roll — every segment header (§8.2) still matches the
identity of the records it contains (sidecar nonce uniqueness, §8.5, is likewise unaffected: `ord`
is journal-global and never repeats across a seam). Replay (§8.7) re-keys to each run-header it
crosses and never attributes a pre-seam record to the incarnation after it.

### 8.2 Segment layout (normative)

Resolution of OQ-12: **per-segment BLAKE3 chaining plus per-record checksums.**

```
segment file = header || record* || seal-record(optional, on clean roll)

header  = magic "DVHCJRN2" (8 bytes)
        || u32-LE format_version (= 1)
        || prev_segment_blake3 (32 bytes; all-zero for the first segment)
        || u32-LE len || header-body-CBOR || u32-LE CRC32C(header-body-CBOR)

header-body = {
  "run_id": bstr .size 32,  "epoch": uint,  "role": tstr,  "instance": uint,
  "module": bstr .size 32,  "segment": uint,          ; segment ordinal, 0-based
}

record framing = u32-LE len || record-CBOR || u32-LE CRC32C(record-CBOR)
record-CBOR    = journal-record                        ; the §8.3 grammar; ord: per-journal
                                                       ; monotone ordinal
```

- **Chaining**: each segment header carries the blake3 of the *complete previous segment file*
  (header, all records, and its seal if any); the final `seal` record (§8.3 tag 17) of a cleanly
  rolled segment carries the blake3 of its own segment's bytes **from the start of the file up to
  but excluding the seal record's own framing** — the seal hash never covers itself (no
  self-reference). This chain is the substrate the Phase D record archive signs.
- **Crash recovery**: on open, the host scans records validating length + CRC32C; the first
  torn/corrupt frame truncates the journal there (the tail is discarded). Recovery then reconciles
  the durable seq counter and spool per §8.4.
- Segments roll at a host-configured size/record threshold; a rolled (sealed) segment is immutable
  and content-addressable.

> **Forward note (non-normative; design-intent, 2026-07-17).** The whole-file BLAKE3 chain +
> per-record CRC32C here is the *base* discipline and is unsigned at base by design (VP-11; OQ-12).
> A reserved, not-yet-built, **generic** verifiability/audit layer (architecture spec §14,
> "Verifiable and auditable training runs") sits **above** this journal without altering it: for
> the segments that carry training inputs / stage outputs it builds a per-segment **Merkle tree
> over record digests** (the per-record BLAKE3 / det-lane digests already recorded here — plus,
> on execution-plane channels, worker-signed proof-native tensor commitments) and extends the
> Phase-D record-archive AttestedHead (§15) from coordinator-only to worker training journals to
> sign that **root** (not the whole-file hash). That enables O(log n) inclusion proofs for
> optimistic dispute resolution (bisect to one step; re-execute against committed inputs on the
> §8.7 replay substrate, or check a single-step validity proof) and, where any verifier-role
> decision procedure is decentralized, mandatory bounded-lag validity proofs of its published
> transition function over the same committed roots. This note changes nothing in §8: the base
> chain, CRC32C, and OQ-12's resolution stand.

### 8.3 The journal record grammar (normative, machine-valid, complete)

The complete record set is **one tagged-union CDDL grammar**. The grammar below is the normative
artifact: it lives verbatim in `daemon-vhc-abi`, it MUST validate as-is under a CDDL validator
(cddl-cat), and **tier-1 CI validates every record of every conformance-run journal against it**
(§13) — a journal containing a record the grammar rejects fails the gate. Tags are permanent; the
union is additive by minor (tags 18–63 reserved). Bodies are canonical-CBOR maps with explicit
string keys; every enumeration is an assigned numeric value.

```cddl
journal-record = run-header-rec / event-rec / read-back-rec / clock-rec / publish-rec
               / timer-arm-rec / timer-cancel-rec / drop-rec / throttle-rec / terminal-rec
               / snapshot-rec / init-rec / signed-frame-rec / instantiation-rec
               / completion-rec / device-profile-rec / condition-rec / seal-rec

hash32 = bstr .size 32
ord    = uint                        ; per-journal monotone record ordinal

run-header-rec = [0, ord, {
  "run_id": hash32, "epoch": uint, "role": tstr, "instance": uint, "module": hash32,
  "abi": uint,                       ; negotiated (major << 16) | minor
  "worlds": { * tstr => uint },      ; negotiated per-world minors
  "bridge": bool,
  "manifest": bstr, "config": bstr, "grants": bstr, "claim": bstr,
  "channels": bstr, "device": bstr,  ; each: verbatim canonical-CBOR bytes of the admitted value
  "format": uint,                    ; journal format version (1)
}]

event-rec = [1, ord, {
  "at": uint,                        ; logical delivery time (§6.5), sampled before delivery
  "frame": bstr,                     ; the exact frame bytes returned by next_event
}]

read-back-rec = [2, ord, {
  "src": uint, "kind": uint, "status": uint,
  ("value": bstr //                  ; inline iff plaintext <= READBACK_INLINE_MAX (§8.5)
   "sidecar": sidecar-ref),          ; else an encrypted content-addressed sidecar (§8.5)
}]

clock-rec        = [3, ord, { "now": uint }]

publish-rec = [4, ord, {
  "channel": uint, "seq": uint,
  "hash": hash32,                    ; blake3 of the guest payload
  "frame": bstr,                     ; the COMPLETE signed wire frame (§8.6, §12)
}]

timer-arm-rec    = [5, ord, { "id": uint, "delay": uint, "armed_at": uint }]
timer-cancel-rec = [6, ord, { "id": uint, "status": uint }]

drop-rec = [7, ord, {
  "class": uint,                     ; 0 = payload-ready, 1 = timer, 2 = gossip, 3 = budget
  "rule": uint,                      ; coalesce code (§2.3): 0 dedup-hash, 1 latest-wins, 2 drop-oldest
  "dropped": drop-id,
}]
drop-id = {
  ? "hash": hash32, ? "timer_id": uint, ? "channel": uint, ? "sender": hash32, ? "seq": uint,
}

throttle-rec = [8, ord, { "paused": bool, "duty_pct": uint, "vram_cap_bytes": uint }]

terminal-rec = [9, ord, {
  "kind": uint,                      ; 0 = outcome, 1 = trap, 2 = forced interruption
  ? "outcome": uint,                 ; present iff kind = 0
  ? "trap": trap-info,               ; present iff kind = 1 or 2
}]
trap-info = { "code": tstr, "import": tstr, "context": tstr, "detail": tstr }

snapshot-rec = [10, ord, { "manifest": bstr }]   ; verbatim accepted state-manifest bytes (§10.2)

init-rec = [11, ord, {
  "config_hash": hash32, "grants_hash": hash32,  ; MUST equal blake3 of tag-0 config/grants bytes
  "status": uint,                                ; the da_init return (§9.4 step 11)
}]

signed-frame-rec = [12, ord, {
  "channel": uint, "seq": uint, "sender": hash32,
  ("frame": bstr //                  ; inline original signed wire frame (Phase A)
   "evidence": evidence-ref),        ; archive reference once durably archived (Phase D)
}]
evidence-ref = { "hash": hash32, "locator": tstr }

instantiation-rec = [13, ord, {
  "counter": uint,                   ; the generation seed of §7.1
  "reason": uint,                    ; 0 = initial, 1 = trap-restart, 2 = upgrade-activation
  "at": uint,                        ; logical time of instantiation (the da_init now() value, §6.5)
}]

completion-rec     = [14, ord, { "op": uint, "result": bstr }]   ; reserved (Phase B)
device-profile-rec = [15, ord, { "profile": bstr }]              ; reserved (Phase B)
condition-rec      = [16, ord, { "code": tstr, "detail": tstr }] ; §6.7 run conditions

seal-rec = [17, ord, {
  "segment_blake3": hash32,          ; hash of this segment EXCLUDING this seal record (§8.2)
  "records": uint,
}]

sidecar-ref = { "hash": hash32, "size": uint, "seg": uint }      ; §8.5
```

`read_back`/`next_event` `NeedCapacity` exchanges have **no record type by design**: the retry is
protocol-mandatory (§4.1, §6.4) and the required length is a deterministic function of the
recorded value, so a record would encode nothing replay needs (rationale recorded per
ratification).

### 8.4 Durability: commit barriers (normative)

Flush ≠ durability. The journal distinguishes **written** (buffered to the OS) from **committed**
(fdatasync of the segment file completed; on segment creation, also the directory entry). Rules:

1. Journal writes are strictly ordered by `ord`; a commit barrier covers every record written
   before it.
2. A commit barrier MUST complete before:
   - **any `publish` returns to the guest** — covering, in one atomic batch: the durable seq
     counter advance, the tag-4 record, and the durable-spool insertion (resolution of the
     publish-atomicity requirement). The seq counter and spool MUST be recovered from / reconciled
     against the journal: on crash recovery, a spooled frame with no tag-4 record is discarded; a
     tag-4 record with no spool entry is re-inserted from the record's `frame` bytes; the seq
     counter resumes strictly above the highest committed tag-4 `seq` (never reused — §12);
   - a `terminal` record (tag 9) is reported to the node;
   - a segment rolls (the `seal` record commits with its segment);
   - a `snapshot` record (tag 10) is acknowledged to the upgrade transaction (§10.3).
3. Inbound-observation records (tags 1–3, 5–8, 12, 16) MAY remain written-but-uncommitted between
   barriers, and the host SHOULD batch their commits. This is safe because an uncommitted
   observation can only be lost together with everything after it: recovery truncates to the last
   committed barrier, and since every externally visible guest effect (a publish) forces a barrier
   covering all prior records, no external effect can ever exist whose journaled cause was lost.
   The post-crash instance restarts from the truncated journal with a fresh instantiation counter.
4. The event pump MUST **write** (not necessarily commit) each event's records before delivering
   the event to the guest thread (§11.4).

### 8.5 Large values: encrypted content-addressed sidecars (normative)

`READBACK_INLINE_MAX = 4096` bytes (an ABI constant in `daemon-vhc-abi`). A `read-back` value whose
plaintext exceeds this is stored as a **sidecar**: a separate file named by the blake3 of its
plaintext, referenced from the record as `sidecar-ref` (§8.3). Sidecars can contain private model
state, activations, or corpus-derived bytes, so they are encrypted at rest under this concrete
profile:

- **AEAD: XChaCha20-Poly1305.**
- **Key scope: one key per journal** (per run-instance journal, generated fresh at journal
  creation), held **node-locally in the node's existing secret storage** (`daemon-credentials`),
  and NEVER written to the journal or any sidecar. One-key-per-journal is chosen over
  per-segment-epoch rotation deliberately: the key never leaves the node's secret store, sidecar
  confidentiality is against at-rest disclosure (not against a live node compromise, which loses
  everything regardless), and journals are bounded per run-instance — so per-journal keys already
  give natural rotation at run granularity without a key-schedule to manage or reference from
  segment metadata.
- **Nonce construction (exact):** the 24-byte XChaCha20 nonce is
  `LE64(ord) || LE64(instantiation counter) || LE64(0)` where `ord` is the referencing record's
  journal ordinal. Ordinals are journal-global and monotone and the key is journal-scoped, so a
  `(key, nonce)` pair is never reused; the instantiation counter is included as belt-and-braces
  against ordinal reuse after an unnoticed truncation.
- **File layout:** `magic "DVHCSC01" (8 bytes) || u32-LE len || sidecar-header (canonical CBOR)
  || ciphertext || 16-byte Poly1305 tag`, with the header as **AAD**:

```cddl
sidecar-header = {
  "run_id": hash32, "epoch": uint, "role": tstr, "instance": uint, "module": hash32,
  "ord": uint,                      ; referencing record ordinal (nonce input)
  "hash": hash32,                   ; blake3 of the PLAINTEXT (the content address)
  "size": uint,                     ; plaintext size
}
```

  On read: verify the AEAD tag (header as AAD), decrypt, then verify the plaintext blake3 against
  `hash`. The execution identity in the header (§8.1) makes sidecar ownership explicit and
  prevents cross-journal splicing (a spliced sidecar fails AAD verification under the owning
  journal's key).
- **Retention and access policy hooks**: sidecars share the journal's retention horizon and are
  garbage-collected with their referencing segments; access is restricted to the machine owner and
  to replay/audit tooling operating under the owner's grants (the decryption key never leaves the
  node's secret store, so any off-node audit requires an explicit owner-mediated export).
  Publishing a journal (e.g. for third-party audit) does NOT automatically publish sidecars; a
  journal consumer without a sidecar (or without the key) sees a typed `ReplayMissingPayload`
  outcome (§8.7), not silent divergence.

### 8.6 Evidence: original signed frames (normative)

The normalized tag-1 `event` record alone cannot back the public record archive or equivocation
evidence — it lacks the signature and the signed envelope. Therefore: for every **authoritative**
`Frame` event delivered, the host MUST also journal the **complete original signed wire frame**
(tag 12), either inline (Phase A — no archive exists) or as a content-addressed evidence reference
`{hash, locator}` once the frame is durably archived (Phase D). Outbound frames are always inline
in their tag-4 record. Advisory/gossip frames MAY be journaled as evidence references or omitted
from tag 12 entirely (their tag-1 record suffices for replay; they carry no consensus weight).

### 8.7 Replay semantics (normative)

Three replay tiers (architecture §3.6); this document fixes the **input-replay (exact)** tier:

- **Input replay MUST be bit-exact on decisions.** Re-feeding a journal — the recorded event frames
  plus the recorded nondeterministic import results — to the same module blob (via the host
  runtime, never the SDK; `host/daemon-vhc-observe`) MUST reproduce every *decision* bit-for-bit:
  every `publish` (channel + payload bytes + resulting seq), every `set_timer`/`cancel_timer`,
  every `read_back` request `(src, kind)`, every branch. Kernels are not re-executed; their
  recorded results are replayed. Handle values reproduce exactly (§7.1 generation determinism).
- The replay verifier serves recorded frames and recorded import results in recorded order and
  **asserts** the guest's outbound actions match the recorded ones, pinpointing the first
  divergence (the existing `ReplayDivergence` shape generalizes).
- **Missing referenced content** (a content-addressed payload that cannot be fetched, or a sidecar
  the consumer lacks): replay MUST fail with the typed outcome **`ReplayMissingPayload`**,
  identifying the hash and the journal ordinal that needed it. Replay up to that ordinal MAY be
  reported for diagnostics, but the run MUST be reported as **incomplete — never as a pass**.
- **Recorded terminal faults** (epoch traps, forced interruptions — tag 9 kinds 1–2): replay
  re-drives the guest from recorded observations up to the recorded ordinal, then **injects the
  recorded terminal fact as the replay outcome** (resolution of OQ-8). No wall-clock mechanism is
  re-armed; wall-clock behavior is never reproduced, only its recorded consequence.
- A journal replays identically on any conforming verifier at the same negotiated tuple; the
  verifier's own budgets play no role (§5.5).
- The journal-soak invariant (refactor invariant 6) activates with A1: every tier-1/2 run records a
  journal and the input-replay verifier runs against it in CI.

---

## 9. `claim()`, the two-instance model, and admission

> **The `claim()` surface below is superseded at the certification minor** by §17.2's
> `da_resource_plan` and `da_apply_execution_grant`. The **two-instance model of this section
> survives unchanged** — it is the assessment/run split, not the claim schema, that the resource-model
> amendments left alone.

> ### SUPERSEDED IN PART — 2026-07-26
>
> **The two-instance model SURVIVES.** A capability-free **assessment instance**, separate from the
> **run instance**, remains the ratified shape — it now serves **`da_resource_plan`** rather than
> `da_claim`. Everything this section says about the assessment instance's isolation, its restricted
> grant set and its cheapness continues to hold.
>
> **The `claim()` export and its tiered-envelope schema are superseded at ABI major 2 minor 5**, per
> `[RC-11]` and `[RC-12]` of the ratified amendments. A guest declares **no device physics**: it
> exports `da_resource_plan`, emitting a backend-neutral **Logical Resource Plan**; the host composes
> `PhysicalEstimate = compose(plan, authenticated Backend Execution Profile)` — a conservative
> estimate that refuses cheaply and sizes the enforced budget, never a proof of fit (architecture
> `[RC-15]`) — and validates it against a measured **Device Capability Report**; the bound
> configuration returns through **`da_apply_execution_grant`** as a logical **ExecutionGrant**.
> `da_claim` retains its meaning only for modules at a lower minor.
>
> **The body below is unrevised and is left that way deliberately.** The minor-5 normative surface is
> the host wave's to write, in the canonical tracked copy `docs/specs/vhc-module-abi-spec.md`. Read
> the text below as the record of the pre-amendment contract, and the ratified amendments
> (**Revision 10** + **§12**) as controlling.

### 9.1 `claim()` (normative)

Every major-2 module MUST export `da_claim` (§2.1). It reports a **tiered memory envelope** as a
deterministic, cheap, compute-free function of the config and grants the module was actually given
(architecture §3.5). The SDK derives most of it from the model definition, replacing v1's host-side
`MetaReport` probe — with a module-owned loop there is no "one step" to dry-run, and none is needed.

```cddl
memory-claim = {
  "hard_accountable": tier-bytes,  ; resources the host meters EXACTLY — the enforceable cap
  "declared_peak":    tier-bytes,  ; expected high-water mark (admission vs owner policy)
  "workspace":        tier-bytes,  ; host-side costs the module cannot see and is not blamed for
  "under_pressure":   [* uint],    ; ordered degradation steps: 0 = deny-new-buffers,
                                   ;   1 = trap-current-slice; 2–15 reserved (additive by minor)
  ? "notes":          tstr,
}

tier-bytes = { "device": uint, "host": uint }   ; bytes
```

Semantics (architecture §3.5): `hard_accountable` is the enforceable cap (breach = `BudgetMemory`/
`BudgetHandles`, attributable); `declared_peak` is judged at admission against owner policy and the
device profile (not hard-enforced; a native-allocator OOM outside metered allocations is absorbed by
the subprocess fault domain, not charged to the module); `workspace` is never blamed;
`under_pressure` is applied by the host in declared order before force-terminating. `claim()` makes
OOM *contained and attributable where possible*, not impossible.

### 9.2 The two-instance model (normative)

Resolution of blocking fix 2. wasm static imports must all resolve at instantiation, so "no
capability imports linked" cannot mean an instance with missing imports. Instead the **same
compiled module is instantiated twice**, with different import bindings:

1. **The assessment instance.** Every capability import — all of `vhc@2`, `net@2`, `sys@2`,
   `data@2`, `compute@2`, and the `tabi@1` bridge — is bound to a **deterministic deny-on-call
   stub**: linking succeeds; *calling* any stub traps **`ClaimCapabilityDenied`** (§7.6), which is
   an admission refusal for this module (a module whose `da_manifest`/`da_claim` needs a
   capability is nonconforming by §2.3/§9.1). The assessment instance runs under minimal fuel and
   a tight epoch deadline; exhaustion traps `BudgetFuel`/`BudgetEpoch` and refuses admission. Only
   `da_abi`, `da_alloc`/`da_free`, `da_manifest`, `da_claim` (and optionally `da_defaults`) are
   called on it. It is used for §9.4 steps 3–7 and then **discarded — never reused, never
   promoted**.
2. **The run instance.** A fresh instantiation with the real capability providers, created only
   after owner authorization (§9.4 step 11). It is the only instance on which `da_init` and
   `da_run` are ever called.

Determinism: `da_claim` MUST be a pure function of `(config, grants)`. The host MAY invoke it more
than once (at assess, at join, at each upgrade re-admission — architecture §5.4) and MUST receive
byte-identical results; a mismatch is the admission refusal **`ClaimInconsistent`** (§1.5). The
claim bytes are journaled in the run header (§8.3 tag 0).

### 9.3 The five-stage admission funnel (normative)

Admission is the owner-bracketed funnel of architecture §3.5, ordered by cost, evaluated in this
exact order; no later stage may run before an earlier one passes:

1. **Owner participation policy** — feature enabled, run/registry allowlists, network posture.
   Free and local; nothing past this line contacts a registry or executes guest code.
2. **Lane feature floor** — each enabled `ParticipationLane` (§9.6) declares a device floor decided
   from the device profile alone (the permanent probe, `daemon-vhc-probe`). Below every enabled
   lane's floor: no module fetch, no coordinator contact.
3. **Run pre-screen** — `device profile ≥ max(lane floor, envelope minimums)`, evaluated **before
   the module is downloaded**. Pre-D0, the host-readable device-minimums section is an **additive
   optional field** on the v1 envelope (a new top-level map key beside `[run]`/`[experiment]`/…):

```cddl
device-minimums = {                  ; v1-envelope additive key "device_min"; mandatory in
  ? "gpu": uint,                     ; envelope v2 (D0). 0 = forbidden, 1 = optional, 2 = required
  ? "vram_bytes": uint,
  ? "ram_bytes": uint,
  ? "disk_bytes": uint,
  ? "up_bps": uint,
  ? "down_bps": uint,
  ? "backend_class": [* tstr],       ; e.g. "cuda", "vulkan"
}
```

   **Cell 5 of the mixed-fleet matrix (a v2 module under envelope v1 + this section) is
   interim-supported CONDITIONAL on a standing fixture test** proving all three provisos
   (ratified conditionally — decisions D3):

   1. the **old** (pre-A2) `FrozenEnvelope::open` accepts and signature-verifies the new raw
      envelope bytes carrying `device_min`;
   2. the original bytes and their blake3 hash are preserved **end-to-end** through every path
      that stores or forwards the envelope (hash computed over received bytes, never re-derived
      from a re-encode);
   3. **no code path decodes the envelope into the old typed `Envelope` and re-freezes
      (re-encodes) it** — the typed struct silently discards unknown fields, so a decode→re-freeze
      round-trip would strip `device_min` and change the hash. This proviso is called out
      explicitly: it is the failure mode that provisos 1–2 alone would miss.

   If the fixture cannot be made to pass against the frozen decoder, **Cell 5 is refused until
   D0** and interim v2 runs gate on the lane floor alone; A2 MUST record which branch shipped.
4. **Module fetch + assessment** — fetch the blob, then run §9.4 steps 1–7 (hash verify → compile
   → assessment instance → manifest → claim), checking manifest ⊆ grants ⊆ lane and claim within
   the lane's claim bounds.
5. **Claim-vs-owner resource authorization** — the node client's standing resource grants (VRAM
   caps, duty cycle) judged against the claim. Last, because it needs the claim as input; supreme,
   like every owner decision.

### 9.4 The admission/bootstrap state machine (normative)

Resolution of blocking fix 3. The complete mechanical sequence from blob to running loop. Steps 1–7
implement funnel stage 4; step 8 is stage 5; steps 10–12 are the join. Any failure at step *n*
aborts with the named refusal and executes no later step.

| # | Step | Failure → refusal |
|---|---|---|
| 1 | **Hash + parse**: verify blob blake3 against the envelope pin **before compiling**; compile (validate) the module | `ModuleHashMismatch`; `BadModule` |
| 2 | **Select ABI/linker**: inspect static import namespaces + exports; select candidate driver; derive + validate the compatibility tuple (§1.3 steps 2–3) | `BadModule`; `WorldMinorUnsupported`; `BridgeRetired` |
| 3 | **Instantiate the assessment sandbox** (candidate linker, deny-on-call stubs, §9.2); cross-check `da_abi` against the candidate + host support and check required exports (§1.3 steps 5–6) | `BadModule`; `AbiDeclarationMismatch`; `AbiUnsupportedMajor`; `AbiMinorTooNew` |
| 4 | **Write canonical config + grants** (the derived grants document, §2.6) into guest memory via `da_alloc` (outside import context, §2.4) | `AllocFail` → refuse |
| 5 | **`da_manifest`** on the assessment instance → decode | decode failure → `BadModule` |
| 6 | **Validate manifest + ABI echo**: `abi` field = step-3 result; static imports ⊆ declared worlds; worlds/custom_ops/channels/events/buffers ⊆ grants ⊆ lane ceilings | `AbiDeclarationMismatch`; `GrantsExceedLane` |
| 7 | **`da_claim(config, grants)`** in the assessment sandbox; validate against lane claim bounds; optionally re-invoke and compare byte-identity | `ClaimExceedsPolicy` (lane); `ClaimInconsistent`; `ClaimCapabilityDenied`/budget traps → refuse |
| 8 | **Owner authorization**: claim vs the owner's standing resource policy (funnel stage 5); **pin `blake3(config)`, `blake3(grants)`, and the claim result** (§2.6) | `ClaimExceedsPolicy` (owner) |
| 9 | **Discard the assessment instance** (unconditionally; also on any earlier failure) | — |
| 10 | On **JoinRun**: re-derive config+grants and verify against the step-8 pinned hashes; if either differs, restart from step 1. On match, re-run **only** step 8 (owner authorization against the recorded claim), then instantiate the **run instance** with real capability providers; journal the run header (tag 0, carrying the full execution identity §8.1) and instantiation record (tag 13) | hash mismatch → restart at step 1; instantiation failure → `BadModule` |
| 11 | **`da_init(config, grants)`** with **byte-identical copies of the admitted bytes**, verified against the pinned hashes (journaled as tag 11; the hashes MUST match the tag-0 header). Nonzero return: journal, tear down, refuse the join | init failure → typed join refusal carrying the guest status |
| 12 | **`da_run()`** — the loop is live (§3, §4) | traps per §7.6 (runtime, not admission) |

Between steps 8 and 10 arbitrary time may pass (assess vs join are distinct node commands —
`AssessRun`/`JoinRun` in today's protocol). The assess→join byte-identity contract — hash pinning,
the cheap step-8-only re-check, and the deliberate absence of a signed assessment token — is §2.6.

### 9.5 Funnel outcomes

| Stage failure | Typed refusal |
|---|---|
| 1 owner policy | not eligible (local; no code, no fetch) |
| 2 lane floor | `Assessed{eligible:false, reason:"below lane floor"}` |
| 3 pre-screen | `Assessed{eligible:false, reason:"device < max(lane,envelope)"}` |
| 4 (steps 1–7) | the §9.4 refusal, carried in `Assessed`/join error |
| 5 owner authorization | `ClaimExceedsPolicy` (owner-policy variant) |

`max(lane floor, envelope minimums)` is the effective floor; an envelope may **tighten** a lane but
never weaken the host's local floor, and a role whose requirements exceed its lane's bounds is
refused at stage 4.

### 9.6 `ParticipationLane` profiles (normative shape; numbers are config)

**This schema is the single canonical `ParticipationLane` definition.** The decisions document's
D7 references it and restates nothing (a prior restatement forked; the fork is resolved by making
this section authoritative). Lanes are versioned host-side node configuration (architecture §3.5);
exact numbers are deployment configuration. Units are **raw `u64` bytes and bits-per-second
exclusively** — no MB/GB/Mbps/kbps field exists anywhere in this schema or in any value derived
from it.

```cddl
participation-lane = {
  "lane":           tstr,            ; "trainer" / "verifier" / "coordinator"
  "version":        uint,            ; bumps only on a breaking field change; additive fields default
  "enabled":        bool,            ; node-side owner switch; reserved lanes ship false
  "worlds":         [* tstr],        ; worlds a role admitted under this lane MAY be granted
  "custom_ops":     [* tstr],
  "bridge_allowed": bool,            ; whether major-2 modules may link tabi@1 under this lane (§2.5)
  "device_minima":  { "gpu": uint,   ; 0 = forbidden, 1 = optional, 2 = required
                      "vram_bytes": uint, "ram_bytes": uint, "disk_bytes": uint },
  "net_storage":    { "up_bps": uint, "down_bps": uint, "disk_bytes": uint,
                      "payload_store_required": bool, "record_archive_required": bool },
  "claim_bounds":   { "device": [uint, uint], "host": [uint, uint] },  ; [min, max] bytes a claim
                                     ; (hard_accountable + declared_peak per tier) must fall within
  "channel_ceilings": { "max_frame_bytes": uint, "spool_frames": uint, "rate_per_min": uint,
                        "per_sender_quota": uint, "replay_window": uint },
  "quiesce_deadline_max_ms": uint,   ; ceiling on the owner-configured drain deadline (§4.4)
  "owner_defaults": { "duty_pct": uint, "vram_cap_bytes": uint,
                      "mode": uint }, ; 0 = always, 1 = idle, 2 = scheduled, 3 = manual
}
```

**Claim-tier → admission/arbitration mapping (normative).** The three claim tiers (§9.1) bind to
enforcement and to the owner's aggregate `ResourceLedger` (decisions D6) as follows — the ledger
MUST reserve **all three**, because charging only the hard-accountable tier can overcommit the
device:

| Claim tier | Enforcement | Ledger treatment |
|---|---|---|
| `hard_accountable` | hard cap; breach is a typed trap (`BudgetMemory`/`BudgetHandles`), attributable to the module | charged (reserved) against aggregate `OwnerBudget` capacity |
| `declared_peak` | not hard-enforced; judged at admission | **reserved** against aggregate capacity (the expected high-water mark occupies budget even before it is touched) |
| `workspace` | never blamed on the module | reserved as **host-computed overhead** (the host MAY substitute its own measured estimate for the module's declared figure; the larger value is reserved) |

Launch profiles (numbers are deployment config; profiles are versioned so a new lane/floor ships as
config, never as an ABI revision): **Trainer** (`enabled: true`, GPU required, the 16/24 GB-class
floor expressed in bytes; all four worlds; `bridge_allowed: true`; the only lane enabled at
launch), **Verifier** (reserved, `enabled: false`), **Coordinator** (reserved, `enabled: false`,
`bridge_allowed: false`). Lanes bound *what a role may require*, not which role labels exist
(architecture §3.5).

---

## 10. Migration scaffolding — `main!`, `snapshot_state`, `da_migrate`

### 10.1 `main!` (guest scaffolding)

The SDK `main!` macro is the v2 analogue of the v1 `experiment!` macro: it emits the required
exports (§2.1) for a type implementing the SDK's module trait, holding the module singleton in a
guest-thread-local (it holds only handles + config, both re-derived through `da_init` after any
re-instantiation). For a driver-shaped module the argument is a driver
(`main!(BarrierRound<TinyLlama>)`, architecture §6); for a hand-written loop it wraps a
`fn(Ctx) -> Outcome`. The macro expands to nothing on non-wasm targets, exactly as `experiment!`
does today.

The scaffolding (macro + snapshot/restore round-trip tests in sim) **lands in the SDK at Phase A**
(refactor §5 A2) so Phase E's upgrade transaction has a tested surface to call; full materialization
of section restore is Phase E.

### 10.2 The snapshot / state-manifest protocol (normative — replaces byte-slice transfer)

Resolution of blocking fix 7. State never crosses the upgrade as one opaque byte-slice through
linear memory. Instead, migration speaks **typed state manifests** — the same shape as the
architecture's checkpoint manifests (§5.3), so checkpointing and migration are one discipline:

```cddl
state-manifest = {
  "schema":   uint,                  ; module-defined state schema version
  "module":   bstr .size 32,         ; producing module hash
  "sections": [* state-section-decl],
}

state-section-decl = {
  "name":    tstr,                   ; e.g. "consensus", "optimizer", "data-cursor"
  "schema":  uint,                   ; per-section schema version
  "hash":    bstr .size 32,          ; blake3 of the section bytes
  "size":    uint,
  "class":   uint,                   ; 0 = consensus-canonical, 1 = role/replica-local (arch §5.3)
}
```

**Producing side (old module).** During a `Quiesce{Upgrade}` drain the guest materializes each
section's bytes host-side with the **Phase-A staging import** (resolution of the migration
bootstrap review):

```
vhc@2::stage_state(ptr: u32, len: u32) -> u64    // staging_id of a sealed, host-staged byte section
```

`stage_state` copies `[ptr, ptr+len)` out of linear memory into a sealed host-staged section and
returns its staging ID. It is prompt (non-blocking), legal during `da_run` (any slice, and during a
drain), and budgeted: staged bytes count against `migration-grant.max_section_bytes` ×
`max_sections` (§2.6) and the per-slice op budget. **Guest-created staging IDs are deterministic**:
they carry the top bit set (`1 << 63`) over a per-instance monotone counter starting at 1, so they
never collide with host-announced (`PayloadReady`) staging IDs (top bit clear) and need no journal
record (the bytes came from replay-reproduced guest memory; the ID is counter-derived). The guest
computes each section's blake3 in-guest (SDK) for the manifest. From Phase B, `buffer.create_from`
handles become an alternative materialization path.

Once every section is staged, the guest calls:

```
vhc@2::snapshot_state(manifest_ptr: u32, manifest_len: u32) -> u32   // 0 = accepted
```

The guest MUST achieve **one successful submission** before returning `QuiesceReady`; rejected
attempts MAY be corrected and retried within the drain deadline (this is deliberately not
"exactly once" — rejection is recoverable). The host verifies each declared section is staged and
hash-consistent, journals the manifest verbatim (§8.3 tag 10) under a durability barrier (§8.4),
and snapshots the sections into host storage. Return statuses: `0 = Accepted`,
`1 = SectionMissing`, `2 = HashMismatch`, `3 = GrantExceeded`; 4–15 reserved (unknown fails
closed, §5.2). A second successful submission in one drain is `BadEvent`. `snapshot_state` outside
a drain is `PhaseViolation`.

**Consuming side (new module).** The host calls `da_migrate` on the **new run instance** after its
re-admission and `da_init` (§10.3 step 4; the §9.4 flow with the tag-13 instantiation record
written **at instantiation, before `da_init`/`da_migrate`** — never deferred to activation).
`da_migrate` receives the host-produced **migration descriptor** — never section bytes, never old
linear memory:

```cddl
migration-descriptor = {
  "manifest": state-manifest,          ; the old module's accepted manifest, verbatim
  "sections": [* migration-section],   ; same order as manifest.sections
}
migration-section = {
  "name":       tstr,                  ; = the corresponding state-section-decl.name
  "staging_id": uint,                  ; the restore staging ID, DIRECTLY in the descriptor
}
```

Each section's restore staging ID is carried **in the descriptor itself** — the new module needs no
`PayloadReady` events (it is not in `da_run` yet and cannot call `next_event`). The module reads
section bytes on demand through the **explicitly granted restore capability**:
`read_back(staging_id, kind = 3 state-section)`, which is **explicitly legal during `da_migrate`**
(the one exception to `read_back`'s `da_run`-only rule, §6.6) and requires
`migration-grant.restore = true` (§2.6). Restore reads are journaled like any `read_back`
(§8.3 tag 2).

```
da_migrate(descriptor_ptr: u32, descriptor_len: u32) -> u32
```

| Return | Meaning |
|---|---|
| 0 | `Ready` — state reconstructed and validated |
| 1 | `Incompatible` — this module cannot consume this descriptor's manifest (schema/section mismatch) → host rolls back (§10.3) |
| 2–15 | reserved |
| ≥16 | module-defined incompatibility detail; treated as `Incompatible`, carried in the journal |

`da_migrate` runs under an explicit bounded budget (fuel + memory + a migration deadline); exceeding
it traps `MigrateBudget` and the host rolls back. It MUST be deterministic given the descriptor, the
staged sections, and config. A module without `da_migrate` is non-migratable: refusal
`MigrateUnsupported` at upgrade re-admission (§1.5 — an admission outcome, never a trap), and the
run falls back to leave-and-rejoin via checkpoint (architecture §5.4).

### 10.3 The host-enforced upgrade transaction (normative shape; implemented Phase E)

A live upgrade is initiated by the node with the **`switch_module`** command — the upgrade-time
peer of the `AssessRun`/`JoinRun` commands that drive first admission (§9.4). `switch_module` is
only meaningful against a worker session that already holds a **long-lived run instance** to
upgrade: until its session holds such an instance, a worker answers `switch_module` with a **typed
command-unsupported** result (an ordinary typed protocol answer — never a trap, never a worker
crash) and attempts no migration. A `switch_module` whose target module omits `da_migrate` is
refused `MigrateUnsupported` at re-admission (§1.5; step 3 below — an admission outcome, never a
trap). On an accepted `switch_module`, the host runs this transaction:

1. **Quiesce** at the SDK-selected fence: deliver `Quiesce{reason: Upgrade, deadline_ms}` (§4.4).
   Authoritative frames spool; advisory events freeze/coalesce. The old module snapshots
   (`snapshot_state`) and returns `QuiesceReady`. Deadline expiry → `QuiesceDeadlineExceeded`
   (forced epoch interruption, §11.3) → treat as failed quiesce → roll back (step 7) or leave.
2. **Snapshot** is already durable (§10.2 + §8.4 barrier) together with the journal cursor.
3. **Admit** the new module: full §9.4 steps 1–9 re-run (owner-law re-check, `da_claim`
   re-evaluation). **Grant-expanding upgrades fail closed** — the worker exits the run rather than
   silently granting more.
4. **Migrate**: instantiate the new run instance — **the instantiation record (tag 13, reason 2)
   is journaled here, at instantiation, before `da_init`/`da_migrate` runs** (it seeds the new
   instance's generations, §7.1) — then `da_init`, stage the snapshot sections, and call
   `da_migrate(descriptor)` (§10.2) under budget.
5. **Validate**: `da_migrate` returns `Ready`.
6. **Activate locally, atomically**: the instance binding swaps to the already-committed
   transition (no host advances the global chain — it advanced when the upgrade record was
   committed). Spooled frames drain into the new instance.
7. **Roll back** to the snapshot on any local failure before activation (`Incompatible`,
   `MigrateBudget`, claim/grant refusal, failed quiesce), then retry or leave the run. A failed
   local migration never rolls back the chain and never resumes the old epoch.

**Journal across the fence (normative).** The upgrade does not restart the journal. The migrated
instance continues the same log per §8.1: the outgoing incarnation's last durable records are its
snapshot manifest (tag 10) and journal cursor (step 2), and the incoming incarnation opens its span
at the seam with its instantiation record (tag 13, reason 2, step 4) and its own tag-0 run-header,
the per-journal ordinal continuing monotonically across the boundary. One replayable log therefore
spans the whole run, upgrades included.

**Sequences across the fence (normative).** Because the migrated instance is a new, never-reused
incarnation, it opens **fresh** signed-stream counters: on every channel the post-upgrade `publish`
sequence **restarts at 0**, in a `(run_id, epoch, role, instance, channel)` stream that is disjoint
by construction from the retired incarnation's — no sequence is inherited or reused across the
fence. This is the upgrade-fence case of §12.2's never-reused stream scope, consistent with the
fresh per-run signing key the new incarnation carries (§12.1).

**The pre-switch assessment (normative — lands with node-side record consumption).**
`AssessRun` gains an additive `switch_target: Option<{ epoch: uint, new_module: bstr .size 32,
grants_hash: bstr .size 32 }>` — the upgrade-time peer of first admission's assess/join two-step.
When present, the worker assesses the COMMITTED TARGET instead of the genesis-pinned module: it
resolves the target bytes from a worker-side source (the explicit hash-verified module override,
the run's filesystem content plane, the content cache, or the presigned content store — an
unresolvable target is a typed refusal, never a guess), re-derives the grants document (target's
linked worlds ∪ the genesis role grant list) and refuses typed unless it hashes to the committed
record's `grants_hash` anchor, then runs the SAME claim admission funnel over the target with the
**admitted role config carried unchanged** (upgrade records pin module + grants; config carriage
arrives when records carry one — until then the migrated instance initializes with exactly its
predecessor's config, so a config-parsing module survives its own upgrade). The answered tuple
carries the target's module hash and grants anchor, `config_hash` = the hash of the carried
config, the target's re-evaluated claim hash — computed where the module bytes live; the node
never touches them — and incarnation `0` (unassigned: the node mints the post-switch
incarnation, exactly as at first join). The session's own pre-fence checks (§10.3
step 3) re-verify every one of these fail-closed regardless.

**Record consumption at the node (normative — the operator-driven product surface).** The node
consumes a committed upgrade record through its product API (an operator submits the
canonical-CBOR record; idempotent via the operation id) and drives `switch_module` from it. The
record is NEVER trusted as presented: the node re-fetches and re-verifies the frozen genesis,
rebuilds the transition chain from that genesis plus its own durable mirror of previously-consumed
records, and validate-appends the presented record — domain tag, run binding, strictly-monotone
gap-free epoch, hash link, stale-old-module, and the authority threshold, any failure a typed
refusal with the chain untouched. Only then: the pre-switch assessment above, a post-switch
incarnation minted STRICTLY ABOVE the running one (an out-of-band-minted incumbent — e.g. a
seat-lease fencing token — can exceed the counter, so the mint takes a floor), identity
provisioning (key + re-issued certificate) BEFORE the command, then `switch_module`. The record
mirror, the advanced execution identity, and the refreshed admitted tuple persist only on
activation; a pre-fence refusal leaves every durable fact unchanged, and a post-fence exit that
left the run persists no advance (the run-level record stays committed regardless — only this
node's instance left).

---

## 11. Threading and bridging (the real wiring)

The architecture says the guest "runs on a dedicated host thread … so blocking host imports are
natural" (architecture §3.1); the refactor warns that "one parked thread" is NOT today's reality
(refactor §5 A2) — today's worker is a tokio async command loop with the device backend pinned to a
dedicated device thread. This section specifies the real wiring.

### 11.1 The three execution contexts (normative)

A running v2 role-instance spans three cooperating contexts inside the worker subprocess:

1. **The guest thread** — **one dedicated OS thread per live role-instance** (resolution of OQ-14;
   N instances on one host = N guest threads, matching "one sandbox = one role-instance",
   architecture §10). It owns the wasmtime `Store` and runs `da_init`/`da_run`. It is the **only**
   thread that ever calls into wasm, and — per §11.3 — the only thread that ever drops the
   `Store`. It blocks synchronously inside `next_event`/`read_back`. Pooling-allocator instance
   limits bound N; exceeding them is an admission-time capacity refusal, not a runtime failure.
2. **The async worker runtime** — the tokio runtime that owns all real waiting: network transport,
   payload/artifact fetch (Phase B+), timers, the framed-stdio node↔worker protocol, and the
   journal writer. It never calls into wasm.
3. **The pinned device thread** — the existing dedicated CUDA/device thread the backends require
   (bridge ops now; `compute@2` in Phase C). It executes tensor ops and signals fences. It never
   calls into wasm.

### 11.2 The bridge between contexts (normative)

- **Guest ↔ async runtime.** `next_event` is a blocking receive on a bounded in-process channel
  from the session's event pump. The pump classifies, orders (§4.7), coalesces (§4.7), and
  journals (§8.4 rule 4) each event, then hands it to the guest thread, which copies it into the
  guest-provided buffer (§4.1). Non-blocking imports (`publish`, `set_timer`, `cancel_timer`,
  `now`) synchronously produce their journaled result (the stamped seq after its durability
  barrier, the timer ID, the cancel status, the clock reading) and enqueue any asynchronous work
  to the runtime.
- **Guest ↔ device thread.** `read_back` parks the guest thread on a completion — from the async
  runtime at Phase A (staged payloads), from the device thread once `compute@2` lands (post-fence
  readbacks). Bridge tensor ops dispatch to the device thread exactly as the v1 dispatch layer
  does today.
- **Backpressure and liveness.** The pump's channels are bounded (§4.7): a full authoritative
  spool back-pressures the network reader (never drops); full advisory queues coalesce per rule,
  journaled. Parking inside `next_event`/`read_back` disarms the epoch watchdog (§5.6); it re-arms
  when a slice starts.

### 11.3 Interruption and teardown (normative)

Adopted correctness fix. There is exactly one interruption mechanism and one teardown owner:

- **Cooperative first.** Throttle/pause and upgrade both begin with cooperative delivery:
  `Budget` (throttle parameter changes) or `Quiesce{Throttle | Upgrade, deadline}` (§4.4). A
  cooperating guest drains and returns from `da_run`.
- **Epoch interruption second.** On deadline expiry (or an immediate owner kill), the host fires
  **epoch interruption** — the only sanctioned cross-thread signal (wasmtime's
  `increment_epoch`/deadline machinery is thread-safe by design). The trap
  (`QuiesceDeadlineExceeded` / `BudgetEpoch`) surfaces **on the guest thread**, which unwinds out
  of wasm.
- **Guest-thread-owned teardown.** The wasmtime `Store` (and with it the instance, its handle
  table, and its device allocations) is dropped **only by the guest thread**, after the trap or
  return unwinds. No other thread may drop or poison the `Store`; cross-thread teardown is
  forbidden. The async runtime observes teardown completion via the guest thread's exit, then
  reclaims host-side resources (spools persist; masters persist per throttle semantics — the
  existing `Command::Throttle{paused}` contract: instance and GPU allocations dropped, CPU masters
  kept). Preemption is churn.

### 11.4 Determinism obligation of the bridge (normative)

- The event pump MUST write each event's journal records before delivering the event to the guest
  thread (§8.4 rule 4); durability barriers per §8.4 rule 2.
- The order in which the pump interleaves classes is a host decision, but it is **recorded** (the
  journal is the delivered order), so replay reproduces it regardless of how a live schedule
  interleaved async wakeups. Two production runs MAY interleave differently; each replays exactly
  as it ran.
- No wall-clock value reaches the guest except through the journaled logical `now()` (§6.5); no
  host-random value reaches the guest at all (§3.3, §7.1).

---

## 12. The signing oracle, domain separation, and channel-scoped sequences

Full semantics are architecture §4.3; this section fixes the ABI-visible surface, because `publish`
(§6.2) is the guest's sole door to it.

### 12.1 The Phase-A signed-frame schema (normative — lands at A2)

Draft 2 granted Phase-A sequences full equivocation semantics while deferring the domain-separated
envelope to D1 — a contradiction the adopted review resolves in the only way consistent with the
"no interim meaning" rule: **the minimum domain-separated signed-frame schema, carrying exactly the
scope-tuple fields, ships in A2 (Phase A).** The evidentiary meaning of a sequence number exists
from the first frame ever signed because the fields that give it that meaning are in every frame
from A2 onward.

```cddl
signed-frame = [envelope: frame-envelope, payload: bstr, sig: bstr .size 64]

frame-envelope = {
  "domain":   tstr,             ; the domain-separation tag; "daemon-vhc/frame/2" at major 2
  "run_id":   bstr .size 32,    ; §8.1 execution identity …
  "epoch":    uint,
  "role":     tstr,
  "instance": uint,
  "module":   bstr .size 32,    ; … ends here
  "sender":   bstr .size 32,    ; the signing identity (ed25519 public key)
  "channel":  uint,
  "seq":      uint,
  "payload_hash": bstr .size 32,   ; blake3 of `payload`
}
```

The signature is ed25519 (`daemon-vhc-proto::sign`: canonical CBOR, `verify_strict`) over the
canonical encoding of `frame-envelope` (which commits to the payload via `payload_hash`). The
module authors **only the payload**; every envelope field is host-built and beyond guest
influence. Receivers MUST verify `sig`, `payload_hash`, and the domain tag before delivering the
frame as a tag-1 event; the verified original is journaled per §8.6.

**Certified per-run keys layer around this frozen envelope** (they ship and are extended by the
certified-identity work; see §12.3). They add the certificate chain that authenticates `sender` to
a base identity and `Authority` semantics over records. **Nothing may add, remove, or change any
`frame-envelope` field defined above** — the fields that give a Phase-A sequence its meaning are
frozen at A2. (Certificates and revocations travel beside the frame or via separate distribution
records, never inside `frame-envelope`.)

### 12.2 Sequence scope and equivocation (normative)

- **Sequence scope:** a sequence number is scoped to the **signed stream**
  `(run_id, epoch, role, instance, channel)` — the execution identity (§8.1) minus `module_hash`
  (which is a function of `(run_id, epoch, role)` via the transition chain and is carried in the
  envelope for attribution) — with each stream owning its own durable, monotone,
  rollback-protected counter (recovered per §8.4 rule 2 — never reused, including across crashes
  and trap-restarts: the tag-13 instantiation counter does NOT reset the stream). **Stream
  identities are themselves never recycled**: because `instance` is the never-reused monotonic
  incarnation id of §8.1 (not a reusable slot), a new role-instance always opens a fresh
  `(run_id, epoch, role, instance, channel)` stream and can never inherit a retired incarnation's
  counter — this non-reuse is what makes the "seq never reused" guarantee sound across the full
  lifecycle, not just within one incarnation. Gap detection (§4.7) and dedup operate on the full
  tuple `(run_id, epoch, role, instance, channel, seq)`.
- **Equivocation evidence** compares the complete scope: two signed frames sharing
  `(run_id, epoch, role, instance, channel, seq)` with different content are self-contained,
  portable, third-party-verifiable evidence (architecture §4.3) — both envelopes carry every scope
  field, so the comparison needs nothing outside the two frames. Frames on different channels
  never collide, and advisory-channel behavior (drops, coalescing) cannot manufacture or mask an
  authoritative gap — the scopes are disjoint by construction.
- **Final semantics from Phase A (OQ-15):** the seq returned by `publish` carries this full
  evidentiary meaning from the first Phase A build. No interim interpretation exists that later
  phases reinterpret.

### 12.3 Certified per-run keys and revocation (normative)

Certificates and revocations are **separate distribution records**, never `frame-envelope` fields
(§12.1). Each carries its own domain-separation tag so no signature is replayable across record
kinds: `daemon-vhc/frame/2` (frames, §12.1), `daemon-vhc/cert/2` (certificates),
`daemon-vhc/revocation/2` (revocations).

```cddl
run-key-cert = {
  "domain":  tstr,            ; "daemon-vhc/cert/2"
  "scope":   cert-scope,      ; the full execution identity the key is bound to
  "run_key": bstr .size 32,   ; the certified per-run public key (== the §12.1 `sender`)
}                             ; distributed as { body: run-key-cert, base_identity: bstr .size 32,
                              ;                  sig: bstr .size 64 } — sig by base_identity over body

cert-scope = {
  "run_id":      bstr .size 32,   ; §8.1 execution identity …
  "epoch":       uint,
  "role":        tstr,
  "instance":    uint,
  "module_hash": bstr .size 32,   ; … the pinned module at this epoch
}

run-key-revocation = {
  "domain":      tstr,            ; "daemon-vhc/revocation/2"
  "run_id":      bstr .size 32,
  "role":        tstr,
  "instance":    uint,
  "revoked_key": bstr .size 32,
  "sequence":    uint,            ; monotonic per (run_id, role) — replay protection
}                                 ; distributed as { body, base_identity, sig } like the certificate
```

- **[CERT-1] Binding.** A certificate binds a per-run key to the **full execution identity**
  `(run_id, epoch, role, instance, module_hash)` (§8.1). A receiver accepts a frame's `sender`
  only when a certificate chaining to a trusted base identity covers that frame's complete scope;
  a mismatch on any scope field, or an uncertified sender, is a typed refusal — never a delivery.
- **[CERT-2] Trusted bases.** The base identities a receiver trusts are named by the run's
  genesis/`Authority` configuration, never ambient config.
- **[CERT-3] One epoch per certificate.** A certificate binds exactly one `epoch`. A committed
  epoch change reissues the certificate over the same per-run key (rebinding to the new epoch and
  its module); full key rotation happens only on an incarnation change. There is no validity
  window — expiry is structural (a certificate dies with its epoch/incarnation).
- **[CERT-4] Revocation + supersession.** A base identity may revoke a per-run key with a signed
  `run-key-revocation`; the monotonic per-`(run_id, role)` `sequence` makes a captured record
  non-replayable. Explicit revocation is best-effort; **incarnation supersession is the safety
  floor** — a certificate for a higher `instance` of a `(run_id, role)` slot implicitly revokes
  every lower incarnation, enforced even under partition (incarnations are never reused, §8.1, so
  the ordering is total).

### 12.4 The coordinator seat lease (normative)

A run's coordinator role is claimed through a **signed, fenced seat lease** stored at the
registry. The registry is **untrusted storage**: it stores the signed object and compare-and-swaps
on the fencing token, but it never verifies signatures and never judges authority — every peer
verifies the lease itself. Leases and releases are separate distribution records with their own
domain tags (registry-format `daemon-vhc/<domain>/<semver>`): `daemon-vhc/seat-lease/1.0.0` and
`daemon-vhc/seat-release/1.0.0` — a lease signature is never replayable as a certificate, frame,
or release, and vice versa.

```cddl
seat-lease-body = {
  "domain":                tstr,           ; "daemon-vhc/seat-lease/1.0.0"
  "run_id":                bstr .size 32,  ; the genesis hash (§8.1)
  "role":                  tstr,           ; the claimed envelope role label
  "epoch":                 uint,           ; the epoch this lease is scoped to
  "incarnation":           uint,           ; §8.1 `instance`, never reused
  "fencing_token":         uint,           ; == incarnation (SEAT-1)
  "claimant":              bstr .size 32,  ; the certified per-run key (== §12.1 `sender`)
  "module_hash":           bstr .size 32,  ; the pinned module the claimant runs
  "endpoint":              control-endpoint,
  "issued_at_ms":          uint,           ; claimant wall clock
  "expires_at_ms":         uint,           ; liveness only, never safety (SEAT-3)
  "heartbeat_interval_ms": uint,           ; advisory renew cadence (TTL >= 3x)
}                          ; distributed as { body: seat-lease-body,
                           ;                  certificate: run-key-cert record (§12.3),
                           ;                  sig: bstr .size 64 } — sig by `claimant` over body

control-endpoint = {
  ? "ws":          tstr,   ; WebSocket control-plane URL
  ? "iroh_ticket": tstr,   ; iroh join ticket
}                          ; at least one member MUST be present

seat-release-body = {
  "domain":        tstr,           ; "daemon-vhc/seat-release/1.0.0"
  "run_id":        bstr .size 32,
  "role":          tstr,
  "incarnation":   uint,
  "fencing_token": uint,           ; == incarnation (SEAT-1)
  "claimant":      bstr .size 32,
}                                  ; distributed as { body, sig } — sig by `claimant` over body
```

- **[SEAT-1] Token ≡ incarnation.** The fencing token is bound to the role-instance incarnation:
  `fencing_token == incarnation`, both carried explicitly, and every verifier asserts the
  equality. A takeover is therefore a new incarnation, which is exactly what advances the
  receivers' supersession floor (§12.3 [CERT-4]) — the registry's CAS and the peers' certificate
  fence advance in step.
- **[SEAT-2] Registry CAS, structural only.** The registry stores one slot per `(run, role)` with
  the current lease and the highest token ever stored (the tombstone floor, which persists across
  release — tokens never reset). It accepts: a first claim at its presented token; a takeover of
  an expired slot (registry clock, plus a bounded skew grace) at exactly `floor + 1`; the holder's
  idempotent refresh at the held token or self-supersession at `held + 1`; a renew whose
  `(run_id, role, incarnation, fencing_token, claimant)` equal the held lease exactly (epoch,
  module, endpoint, and expiry may change — the epoch-rebind rule, [CERT-3]); and a release whose
  token and claimant match. Everything else is a typed structural refusal that mutates nothing.
  The registry MUST validate structure (domain tag, token≡incarnation, expiry window, a dialable
  endpoint, slot consistency) and MUST NOT verify signatures or judge authority.
- **[SEAT-3] Peer acceptance.** A peer accepts a seat lease only when: the claimant's signature
  over the body verifies; the embedded certificate chains to a genesis-trusted base identity
  (§12.3 [CERT-2]) and covers exactly the lease's scope `(run_id, epoch, role, incarnation,
  module_hash)` with `run_key == claimant`; the token≡incarnation invariant holds; the lease is
  unexpired (with the skew grace); and the incarnation is not below the receiver's supersession
  floor or explicitly revoked (§12.3 [CERT-4]). A stale claimant's records are refused once a
  higher fencing token exists, **regardless of what the registry says** — wall-clock expiry gates
  takeover liveness only; fencing is the safety mechanism.
- **[SEAT-4] Transport surface.** The registry exposes the seat over
  `GET`/`PUT`/`DELETE {base}/runs/:id/seat/:role` and
  `POST {base}/runs/:id/seat/:role/heartbeat`, canonical-CBOR bodies both ways; an accepted
  mutation answers 200, a refused one 409 with the structural decision and the slot's current
  state (the loser's re-read). The CAS semantics of [SEAT-2] are pinned by shared test vectors
  (`daemon-vhc-proto/tests/fixtures/seat-cas-vectors.json`); every conforming registry
  implementation reproduces them exactly.

### 12.5 Wire-schema ownership: mechanism vs round vocabulary (normative)

The wire surface splits into two ownership classes, realized as two crate homes (architecture §7
rule 1 — "daemon-vhc-proto stays algorithm-free: no assignment math, no round vocabulary"):

- **[OWN-1] Mechanism — `daemon-vhc-proto`.** The canonical CBOR codec, ed25519 signing
  (`sign`/`verify_strict`, `Signed<T>`), blake3 hashing + content addresses, merkle set
  commitments (`commit_set`/`SetCommitment`), the genesis (schema-2) envelope + freeze/verify,
  grants vocabulary + admitted quotas, run-key certificates + revocations (§12.3), the transition
  chain, capability sets, and `VhcProtoVersion`. Mechanism is what a party that never interprets
  an algorithm's messages still needs: it carries and verifies bytes, it never gives them round
  meaning.
- **[OWN-2] Round vocabulary — `daemon-vhc-sdk-consensus` (the SDK schema layer).** The round
  message schemas (`RoundOpen`, `Commitment`, `Attestation`, `StorageReceipt`, `RoundRecord`,
  `Digest`, `Straggle`, `Join`, `Heartbeat`, `CheckpointAttestation`, the externally-tagged
  `VhcMessage` union and its `SignedMessage` control frame), the round state-digest schedule
  (`derive_schedule`/`digest_state`), the committed-set object (`RecordSet` /
  `record-set.cbor`), and their authoritative CDDL (`daemon-vhc.cddl`, validated by the
  conformance suite that lives with the schemas). Assignment math and the coordinator `tick`
  already lived here; the schemas they consume now do too.
- **[OWN-3] Host opacity (structural).** No production host crate (`daemon-vhc-host`,
  `daemon-vhc-net`, `daemon-vhc-session`'s production modules, `daemon-vhc-node`,
  `daemon-vhc-supervisor`, the `daemon-vhc-worker` binary) may link the SDK schema crates
  (`daemon-vhc-sdk-consensus`, `daemon-vhc-sdk-rounds`): hosts route opaque signed frames and
  content-addressed payload bytes; only modules (guests), SDK layers, and explicitly-exempted
  harness/oracle tooling (testkit, observe, the e2e leaf crate, and `harness`-feature-gated
  optional edges) decode round messages. Enforcement is a dependency-direction gate plus a
  negative architecture test: the resolved default-feature normal graph of every production host
  crate — the shipped worker binary above all — must not contain a schema crate.
- **[OWN-4] Cloud coordinator seat.** The wasm-tick coordinator DO's decision core is the
  compiled `daemon_vhc_sdk_consensus::coordinator::tick` — the same pure state machine the
  node's `coordinator-quorum` guest wraps — and its I/O shell carries a **vendored TS mirror**
  of [OWN-2]'s schemas (`packages/shared/src/vhc` in the cloud repo) for exactly the edges the
  pure tick refuses (availability-evidence injection, record-set persistence). The vendored
  mirror and the compiled tick's provenance (source commit + content hashes) are recorded next
  to the artifact (`coordinator.wasm.provenance.json`); a schema change re-vendors both. The
  coordinator seat is a *module seat*, not a host: carrying round vocabulary there is by design.
- **[OWN-5] Two signed-frame forms (transitional; inventoried for the WireVersion decision).**
  The §12.1 domain-separated `frame-envelope` is the module-facing transport frame the host pump
  verifies and journals. The [OWN-2] `SignedMessage` (`{version, payload, signer, sig}`, signed
  over canonical CBOR of `(version, payload)`, no domain tag) is the control-plane frame the
  WS/registry/coordinator-DO plane and the round vocabulary ride today. They are DISTINCT
  encodings serving one architecture; converging the control plane onto the §12.1 envelope is a
  deliberate, separately-ratified wire change — until then, [OWN-1] owns the §12.1 envelope
  mechanics and [OWN-2] owns `SignedMessage`, and no implementation may treat one as the other.

### 12.6 Role-session lifecycle wire (normative — lands with the role session)

The generic role session (the single worker runtime for any role in any run) adds a lifecycle
surface to the node↔worker event wire and one guest-visible drain reason. All additions are
name-keyed CBOR (additive decode), inventoried for the accumulated WireVersion decision.

- **[RS-1] The generation counter.** Every run-scoped worker event (`RunPhase`,
  `RoundProgress`, `RoundOutcome`, `CheckpointPublished`, `ModuleSwitched`,
  `AdmittedTupleMismatch`, `RunTerminated`) carries `generation: u64` — the never-reused
  role-instance incarnation id (ABI §8.1 `instance`). The node MUST discard a run-scoped event
  stamped with a generation below the run's live incarnation: a reaped instance's late events
  can never mutate its replacement. The field decodes additively (`0` = authored before the
  counter existed; consumers treat `0` as un-stamped, never as stale).
- **[RS-2] The terminal event.** `RunTerminated { run_id, generation, outcome }` is the LAST
  run-scoped event a generation emits — exactly one per spawned role task. `outcome` is the
  classified terminal:

  | Class | Meaning | Node reaction |
  |---|---|---|
  | `Completed { outcome: u32 }` | the module signaled run end (`da_run` returned; `0` = clean) | terminal; release resources; never rejoin |
  | `Left { checkpoint: Option<text> }` | owner intent (leave); on a graceful leave `checkpoint` is the blake3 hex of the drain snapshot persisted to the payload plane | terminal; owner-driven |
  | `FailedRetryable { reason: text }` | recoverable environment fault: transport loss, provider fault, resource-budget breach, an unrecoverable inbound sequence gap with no backfill | may reconverge (rejoin as a NEW incarnation) under the retry budget |
  | `FailedTerminal { reason: text }` | module trap, admission identity mismatch, certificate refusal, init/migrate refusal | no automatic rejoin; owner action |

  `reason` strings are operator-facing detail and MUST NOT be branched on; the class is the
  contract. Terminal handling MUST be idempotent (duplicate delivery cannot double-release).
- **[RS-3] The leave drain reason.** `quiesce-reason` 2 (graceful-leave drain) joins the §4.4
  table beside 0 (upgrade) and 1 (throttle): owner intent ends the role instance; the module
  snapshots at the fence (`snapshot_state`) and the session persists the capture to the payload
  plane as the leave checkpoint. Guest-visible (the reason rides the `Quiesce` event); a module
  with nothing to snapshot returns `QuiesceReady` without a submission, and the terminal `Left`
  then carries no checkpoint.
- **[RS-4] Content-addressed payload keying.** The role session services module
  `payload_put`/`payload_get` and `data.fetch` against a **content-addressed** store seam
  (`payload/<blake3>`): the module names content, the store moves bytes, and the pump re-verifies
  every fetched object against the requested hash before delivery. Run/round/peer coordinates
  never appear in a production payload key (they are round vocabulary); the coordinate-keyed
  store form survives only in harness-era consumers.
- *(informative)* The session's event-driven bridge over the run pump (the egress wake) and the
  hard-pause enforcement point (pump-level delivery freeze while `paused`; duty percentage stays
  a cooperative `Budget` advisory) are host-internal mechanics, not wire surfaces; they are
  recorded here only to delimit this section's wire scope.

### 12.7 The chunk-addressed corpus contract (normative — lands with the data wave)

Training data reaches a module ONLY as verified byte ranges of genesis-pinned, chunk-addressed
artifacts. The host is mechanism (fetch + verify + slice under grants); assignment, windowing,
batching, ordering, and epoch scheduling are module policy (the guest SDK's corpus layer). No
host batch staging exists on the production path.

- **[CC-1] The corpus manifest.** A canonical-CBOR document (`daemon-vhc-proto` writer, so its
  hash is reproducible), format major 1, pinning: token width (`u16`/`u32`) **and endianness**;
  `seq_len`; the sequence-boundary rule (format 1: whole sequences per shard — a sequence never
  spans a shard boundary); optional `eos_id`/`pad_id`; the corpus-wide `chunk_size` (a whole
  multiple of the token width — **no token ever spans a chunk boundary**; ratified default
  4 MiB); the tokenizer identity `{hash, name, revision}` where `hash` is the blake3 of the
  tokenizer artifact itself (content-addressed like any pinned artifact); `total_tokens`; and
  the shard list `(shard_hash, byte_len, token_count, chunk_hashes[])` in data-window order.
  The manifest's blake3 is its identity; the genesis envelope pins it in a `corpus_manifest`
  field that MUST also name a fetchable `[artifacts]` entry — the pin commits the run's data
  identity into the genesis hash (`RunId`).
- **[CC-2] Shard identity is the chunk fold.** `c_i = blake3(chunk_i bytes)` (plain content
  hashes — chunks ride every content-addressed seam: stores, caches, the §8.7 replay payload
  table), and the shard's artifact identity is the domain-separated fold
  `blake3("daemon-vhc/corpus-shard/1.0.0" ‖ u64le(chunk_size) ‖ u64le(token_count) ‖
  u64le(byte_len) ‖ c_0 ‖ … ‖ c_{n-1})` — NOT the blake3 of the shard bytes. Order and geometry
  are committed by the fold; whole-shard verify-on-first-touch is thereby structurally
  impossible for corpus shards (rejected: it defeats streaming), and an unregistered fetch of a
  fold identity can never pass verification by accident. The fold definition is a shared
  **chunk-addressed artifact identity** clause, instantiated per derivation domain: corpus
  shards under `daemon-vhc/corpus-shard/1.0.0` (with `token_count` in the preimage), det-state
  families under `daemon-vhc/det-state/1.0.0` (§12.14 — geometry is `(chunk_size, byte_len)`
  only). The same chunk list under different domains yields unrelated identities.
- **[CC-3] `data@2::register_chunks(desc_ptr, desc_len) -> u32`** (introducing minor 2). The
  module presents one shard's chunk map as canonical CBOR
  `[chunk_size, token_count, byte_len, [c_0, …]]`. The host re-derives the fold and admits the
  map **only when the fold IS a granted artifact hash** — an ungranted fold traps
  `GrantViolation` (a module cannot register chunk identities for content it was not granted); a
  malformed/degenerate descriptor traps typed at the call. Returns 0; re-registration of the
  same identity is idempotent. Registration is deterministic guest output (§2.7 `dc` class): no
  journal record; replay re-executes it over reproduced guest memory.
- **[CC-4] Ranged fetch on a registered identity.** `data@2::fetch(fold, off, len)` keeps its
  §2.2 signature. Bounds are knowable at the call (registration pinned `byte_len`): an
  out-of-bounds range completes `Err(StoreRefused)` immediately. Otherwise the host computes the
  chunk-aligned **covering span** of `[off, end)`, asks its provider for ONLY that span (an
  in-process content store may answer with the whole object; a live store serves an HTTP Range),
  verifies every covering chunk against the REGISTERED chunk hashes, then slices the exact
  requested range into the completion buffer. A lying span (wrong length, wrong bytes) completes
  `Err(HashMismatch)`; the guest never observes unverified bytes. At replay, a chunked fetch
  completion materializes from CHUNK-keyed payload-table entries (a missing chunk is the typed
  `ReplayMissingPayload` divergence).
- **[CC-5] The data-read budget.** The grants vocabulary gains `data-read-budget` (raw bytes,
  cumulative per role instance; `0` = unbounded by this grant), derived tighten-only like every
  §2.6 bound and carried in the role grant list, the lane ceilings, the admitted quotas, and the
  grants document. The host charges each `fetch` at the CALL from the requested range
  (guest-call-order deterministic — never from embedder timing); a breach completes
  `Err(GrantExhausted)`, never a trap and never silent truncation. Refusal completions mint
  their `OpId` through the one `begin()` sequence (§7.1) and journal as ordinary tag-14 records.
- **[CC-6] Assignment/windowing are module policy.** The guest SDK derives a peer's epoch
  assignment deterministically from `(genesis_hash, epoch, roster_size, peer_index)` under the
  batch-assignment domain salt: disjoint stripes covering the corpus exactly once, re-laid every
  epoch (the ratified per-epoch reshuffle). Windowing maps sequence ids to in-shard byte ranges
  and coalesces adjacent fetches; token decode honors the manifest's pinned width AND
  endianness. None of this exists host-side.
- **[CC-7] Retired: the credentials-carried corpus reference.** The engine-era
  `JoinCredentials.EngineParams.corpus` reference object (manifest hash + window, the JSON
  `corpus/<blake3>.json` schema) is REMOVED. The genesis `corpus_manifest` pin is the one corpus
  reference; corpus objects publish as `corpus/<manifest blake3>.cbor`,
  `corpus/<tokenizer blake3>.json`, and `corpus/<fold>.bin`.

### 12.8 Live-transport carriage: distribution records on the plane, plane-selection credentials, content presign (normative — lands with the live transport attach)

The live transport attach adds three wire surfaces. All are additive (name-keyed CBOR /
structurally disjoint shapes), inventoried for the accumulated WireVersion decision.

- **[LT-1] Distribution records travel ON the control plane, structurally disambiguated.** The
  §12.3 certificate and revocation records propagate over the same byte plane as §12.1 frames,
  wrapped in a **distribution record**: a top-level canonical-CBOR **single-entry map** whose key
  names the record kind and whose value is the §12.3 record verbatim:

  ```cddl
  distribution-record = { "cert": run-key-cert-record }        ; §12.3 { body, base_identity, sig }
                      / { "revocation": revocation-record }    ; §12.3 { body, base_identity, sig }
  ```

  A §12.1 frame is a top-level **array** `[envelope, payload, sig]`; the two top-level shapes are
  disjoint, so a receiver classifies structurally (attempt the record decode; hand everything
  else to the frame attach) without speculative decoding of frame bytes. A future record kind is
  a new map key, refused typed by old receivers. Delivery stays best-effort (§12.3 [CERT-4]:
  supersession is the safety floor); a session announces its own certificate at attach and
  re-announces on every reconnect (the WS resubscribe registration).
- **[LT-2] The ingest trust gate.** A receiver ingests a distributed certificate ONLY after the
  record's chain verifies AND its base identity is genesis-trusted (§12.3 [CERT-2]); an
  unverified record MUST NOT advance any trust state — above all the incarnation supersession
  floor, where a forged high-incarnation record would otherwise fence out the legitimate holder.
  Revocations ingest only through the replay-protected ledger (§12.3 [CERT-4]). A refused record
  is a typed per-record advisory, never a session fault; re-delivery of an ingested record is an
  idempotent no-op.
- **[LT-3] Plane-selection credentials.** `JoinRun.credentials` on the role-session path carries
  the node-authored **plane selection** (`session-credentials`): the run's genesis hash (the
  binding + iroh topic input), the WS base + auth mode, an optional PUBLIC iroh half (relay URLs
  + bootstrap roster), an optional presign base, and bootstrap §12.3 peer certificates. The body
  is secrets-free BY CONSTRUCTION: the signing identity, the iroh transport secret, and (with
  the credential-authorship rework, which extends this body additively) all token material are
  keystore references — never command-payload bytes (§12.3 [CI-9]-adjacent custody; the
  engine-era `JoinCredentials` body, which carried a raw signing seed and the iroh secret, is
  retired from this surface). A buffer that is not a `session-credentials` means "no live
  attach": the referenceless smoke seat or a typed refusal — never a silent local run.
- **[LT-4] Content-addressed payload presign — no registry surface change.** The [RS-4]
  content-addressed payload plane rides the FROZEN §11.1 presign contract using the **artifact**
  object form at the run-relative key `payload/<blake3-hex>` (bucket key
  `runs/<run>/payload/<hex>`; the `runs/<run>/` prefix is the store's own namespace from the
  per-run presign endpoint, never a module-visible coordinate). A conforming registry's presign
  surface already accepts `(artifact, put|get)` structurally, so no route, kind, or key rule is
  added; live confirmation against the dev R2 backend rides the R2-conditional lane (the SigV4
  API-token-scope fix), with the mock/filesystem gates mandatory meanwhile.
- *(informative)* The durable per-incarnation journal home (`<run state dir>/<role>-<incarnation>/journal`,
  §8) and the run-state-root path reference the node exports to its worker are node↔worker
  process contract, not CBOR wire; they are recorded with §8's execution-identity text and noted
  here only to delimit this section's scope.

### 12.9 Credential authorship, node-directed identity, and checkpoint carriage (normative — lands with credential authorship)

The node authors every run instance's identity + credentials and delivers the minted incarnation
and any late-join restore to the worker. All wire additions are name-keyed / additive.

- **[LT-5] Node-directed role selection.** `AssessRun` gains an additive `role: Option<text>` —
  the envelope role label the node directs assessment at (the coordinator-seat path directs the
  coordinator role). A label absent from the genesis role set is a typed refusal; `None` is the
  single-trainer default (the first role whose declared LANE is not `coordinator`, never a
  label-substring heuristic).
- **[LT-6] The admitted tuple carries the minted incarnation (populated, not shape-changed).**
  `JoinRun.admitted_tuple` — additive since the role session landed, formerly always absent — is
  now MANDATORY on the production join path and carries the node-minted, never-reused incarnation
  the instance runs as (`incarnation > 0`). The worker rederives the artifact-addressed fields and
  compares field-by-field (a mismatch is the typed `AdmittedTupleMismatch`); a join with no tuple
  is a typed refusal. The incarnation is the generation stamped on every run-scoped event
  (§12.6 [RS-1]).
- **[LT-7] Secrets by reference; the credentials record.** `SessionCredentials` (§12.8 [LT-3])
  gains the additive `secret_ref: Option<text>` and `expires_at_ms: uint`. Token material lives
  ONLY in a node-authored keystore CREDENTIALS RECORD (`{ ws_auth, expires_at_ms }`, canonical
  CBOR, mode-0600, atomically rewritten on refresh); the wire body's `ws_auth` stays `None` and
  carries only the reference ([CI-9] custody — no token on a command payload or in a journal, pinned
  by a byte-scan negative). `vhc.db` persists the `credentials_ref`, never the secret. The worker
  resolves the record against the run's own key directory (the reference is a bare record name — a
  path component is refused). Expiry drives a plane re-resolve, never a session restart.
- **[LT-8] Node-issued per-run certificate; the worker never mints.** The node mints the per-run
  key and issues its `run-key-cert` (§12.3) under the base identity at join authorship; the base
  identity never leaves the node process. The worker resolves the key + certificate READ-ONLY by
  reference and refuses typed when either is absent or the certificate does not bind exactly the
  execution identity about to run. Terminal cleanup (§12.3 [CI-7]) deletes the key, certificate,
  and credentials record on a terminal outcome that ends the run identity (Completed / Left /
  FailedTerminal); a FailedRetryable's material survives for reconvergence.
- **[LT-9] Late-join checkpoint carriage.** `SessionCredentials` gains an additive
  `restore: Option<{ round: uint, hash: bstr .size 32 }>` — the node-resolved registry checkpoint
  pointer the fresh instance migrates from before it runs (§10.2/§10.3 migration input). The worker
  fetches the checkpoint DOCUMENT by content address, hash-verifies it, decodes the snapshot
  (`[manifest, [[section-name, section-bytes], …]]`, canonical CBOR — the same document the drain
  snapshot writes), and refuses typed if it cannot be resolved (never a silent fresh start). The
  checkpoint POINTER rides the frozen registry surface, keyed **per `(role, kind)` slot**: the
  node publishes `POST {base}/runs/:id/checkpoint` (`{role, kind, round, hash, size}`) on the
  session's `CheckpointPublished` — `kind` is `"live"` (the periodic cadence, [LT-11]) or
  `"drain"` (a graceful-leave drain snapshot) — and reads `GET {base}/runs/:id/state`.
  `data.checkpoints` (the slot list) at join. Restore resolution is ROLE-SCOPED with a kind
  preference: the joining role's freshest `live` pointer, else its freshest `drain` pointer,
  never another role's slot — a coordinator drain snapshot can never shadow a trainer restore
  source, and a drain snapshot never shadows a fresher live one. Within a slot the registry keeps
  the freshest round (same-round byte-identical re-uploads cross-check it; a same-round divergent
  hash is rejected).
- **[LT-11] The periodic live checkpoint cadence (normative — lands with hard-crash
  continuity).** A drain snapshot exists only when an instance drains; a hard-crashed peer never
  drains. The trainer module therefore exports its full restorable state (the same sections its
  drain snapshot stages, plus its `round` watermark) on a config cadence (`ckpt_every` ingested
  rounds; `0` disables) and `payload_put`s it as the SAME checkpoint-document shape [LT-9]
  decodes. The session recognizes such a put **structurally over host-owned shapes only** — the
  document envelope paired with the drain writer, whose manifest is the §10.2 state-manifest map,
  every declared section hash-matching its bytes, a `round` watermark section present — and
  surfaces `CheckpointPublished{kind: "live"}`; the module's round vocabulary is never decoded
  (a coincidental or corrupt object cannot register a pointer: the hash discipline rejects it).
  `CheckpointPublished` gains the additive `kind` field (absent decodes as `"drain"`, the
  pre-cadence sole source). A restored peer's continuity across a hard crash follows from
  [LT-9]'s live-preferred resolution: within a replicated trainer group, any peer's fresher live
  checkpoint is a valid restore source for a rejoiner, so the crashed peer resumes from state
  that already folds the rounds it missed.
- **[LT-10] Sealed record-archive publication — content-store objects, no new wire.** A
  coordinator's SEALED journal segments (§8.2, each content-addressed by its complete-file blake3)
  publish to the content-addressed payload plane as ordinary objects; the head (last sealed
  segment) is the archive anchor the offline replay oracle fetches. No registry route or schema is
  added — the archive is content addresses on the existing payload plane.

### 12.10 The durable run-instance state machine (normative — lands with node reconciliation)

The node is the single authority for a run instance's lifecycle. The state machine is TWO-AXIS
and durable in the node's participation store: the **owner-intent** axis (`joined | paused |
left`) records what the owner wants; the **observed** axis (`running | completed |
failed_retryable | failed_terminal | left`) records what the last incarnation did. The app-facing
projection is the six-state view `running | completed | paused | failed_retryable |
failed_terminal | left` (terminal observations win; a paused intent masks recoverable states).
No new node↔worker wire is added — this section consumes §12.6's generation-stamped events and
`RunTerminated` exactly; the deltas below are the node's durable/API contract.

- **[RL-1] Terminal transitions consume `RunTerminated` idempotently and generation-gated.**
  Exactly the §12.6 [RS-2] class table drives the observed axis. A duplicate terminal for an
  already-transitioned instance is a no-op (it can never double-release resources); a terminal —
  or ANY run-scoped event — stamped with a generation other than the run's current incarnation is
  discarded whole (a reaped instance's late events never mutate the replacement, including its
  key custody).
- **[RL-2] Teardown is observed before the ledger releases.** The release order is fixed: (1) a
  durable RELEASE MARKER commits — worker teardown observed (terminal event received, or the
  worker's event stream closed / the process was reaped) with the terminal target recorded; (2)
  the live instance leaves supervision and its resource reservation releases — a replacement is
  never admitted while the predecessor may hold devices; (3) the terminal state commits and the
  marker clears. A node crash between (1) and (3) is repaired by the startup reconciliation pass:
  every child died with the node, so process absence makes teardown definitional and the recorded
  target simply commits — a repaired `completed` is never rejoined.
- **[RL-3] `completed` never restarts; `failed_terminal` requires owner action.** The restart
  reconvergence set is `joined` intents whose observed state is non-terminal; a module run end and
  a terminal failure drop out of it permanently. An explicit owner rejoin of a settled run mints a
  FRESH incarnation (the predecessor's identity, keys, and journal stream are settled).
- **[RL-4] The bounded retry budget.** A recoverable failure consumes one attempt of a
  config-bounded budget with exponential backoff (config-surfaced defaults; the reconvergence
  fires from a periodic reconciliation pass); exhaustion escalates to `failed_terminal` with a
  typed reason. The budget resets when an incarnation stays running past a configured minimum
  uptime (the coarse stability signal — the node never inspects rounds), so a crash loop cannot
  launder its budget by merely restarting. Mid-run reconvergence ALWAYS mints a new never-reused
  incarnation with freshly-authored credentials + certificate (§12.9); only a node restart —
  where no live predecessor can exist — retains the persisted incarnation.
- **[RL-5] Pause is durable owner intent with release-on-pause.** The client API gains
  `VhcPause { run_id, op_id }` / `VhcResume { run_id, op_id }` beside join/leave (op_id-idempotent
  intents; name-keyed CBOR, additive). Pause persists the intent FIRST (a paused run survives
  restart and is never reconverged until resumed), delivers the hard pause lever (§12.6 —
  memory, not just time), releases the resource reservation, and surrenders any held coordinator
  seat lease (the fenced release; the floor persists). Resume re-admits against the owner's
  CURRENT ledgers before lifting the pause — a refusal is typed and loud, and the run stays
  paused with nothing half-claimed.
- **[RL-6] The lifecycle projection on the run summary.** The client run-summary DTO gains the
  additive fields `run_state` (the six-state projection), `retry_count` (consumed budget), and
  `terminal_reason` (the typed reason recorded with a terminal transition; operator-facing
  detail, never branched on). Skip-encoded when absent; a pre-lifecycle producer decodes
  unchanged.
- **[RL-7] Coordinator-seat residency (no new wire).** When the owner enables coordinator duty,
  a resident keeper covers every joined run whose admitted role is the configured seat role over
  the EXISTING §12.4 seat routes: claim when a bid derives (stand by against a live incumbent),
  heartbeat-renew under the same token at the lease cadence, drop the lease when a renew is
  refused (fenced — never a takeover fight), and release signed on owner pause/leave and node
  shutdown so a successor bids floor + 1 without waiting out the TTL.

### 12.11 The live module switch wire (normative — lands with the switch under the role session)

The §10.3 upgrade transaction runs against the role session's held instance. The node initiates
it with `switch_module`; everything below is additive, name-keyed CBOR, inventoried for the
accumulated WireVersion decision.

- **[SW-1] The post-switch admitted tuple.** `switch_module` gains
  `admitted_tuple: Option<admitted-tuple>` — the node-assessed tuple for the POST-SWITCH
  execution identity: `module_hash` = the command's target, `grants_hash` = the committed
  upgrade record's anchor, `config_hash`/`claim_hash` rederived for the target, and —
  load-bearing — the node-minted, never-reused **incarnation** the migrated instance runs as
  (§8.1: a live upgrade advances the epoch AND mints a new incarnation; the incarnation must
  strictly supersede the running one). The worker rederives the artifact-addressed fields from
  the artifacts about to run and refuses any mismatch typed; a production switch without a tuple
  is a typed refusal. The field decodes additively (`None` on a pre-switch-wire frame).
- **[SW-2] The certificate re-issuance handshake (no secret on the wire).** Before sending
  `switch_module`, the node provisions the post-switch identity in the run's identity keystore:
  it mints the NEW incarnation's per-run key (an incarnation change rotates the key, §12.3
  [CERT-3]) and issues its `run-key-cert` bound to exactly
  `(run_id, target epoch, role, new incarnation, new module_hash)`. The worker resolves both
  READ-ONLY by reference ([LT-8] custody) and refuses typed when either is absent, the scope is
  not exactly the post-switch identity, the certified key is not the provisioned key, or the
  chain does not verify. On activation the session re-announces the re-issued certificate as a
  §12.3 distribution record; the retired incarnation's certificate dies by supersession
  ([CERT-4] — the new incarnation advances every receiver's floor).
- **[SW-3] The refusal/terminal split on the answer surface.** A switch refused BEFORE the
  transaction touches the running instance — no live instance held, an unresolvable or
  hash-mismatched target artifact, a tuple/certificate handshake failure, an owner-law
  re-admission refusal or fail-closed grant expansion (all evaluated ahead of the fence; they
  are effect-free) — answers with the new event
  `switch_refused { run_id, generation, reason }`: the old module keeps running untouched and
  the node reassesses/reprovisions. Past the fence (the old instance quiesced), a
  migrate/validate failure that exhausts its bounded rollback-and-retry LEAVES the run per
  §10.3 step 7: the session emits the terminal `run_terminated` (§12.6 [RS-2],
  `failed_terminal`) — never a silent old-epoch resume. Activation answers the existing
  `module_switched`, now stamped with the NEW generation; every subsequent run-scoped event
  carries it (§12.6 [RS-1]).
- **[SW-4] Target-artifact resolution.** `switch_module` carries the target by hash only. The
  worker resolves bytes from, in order: an explicit node/dev-controlled module-source override
  (hash-verified against the command's target — the override substitutes the fetch, never the
  pin), or the session's bound content-addressed stores (`payload/<blake3>`, §12.6 [RS-4]); the
  pin is re-verified wherever the bytes came from. An unresolvable target is a [SW-3] pre-fence
  refusal.
- **[SW-5] The journal seam on disk.** The durable journal home realizes §8.1's continuation:
  the seam SEALS the retired incarnation's segment and rolls to a new segment whose header
  carries the incoming execution identity, in ONE chained file series (the join incarnation's
  journal directory); the per-journal record ordinal stays globally monotone, the incoming span
  opens with its own tag-0 run-header, per-channel publish counters reset (the new §12.2 stream
  opens at seq 0), and crash recovery re-keys its durable counters at each run-header so a
  post-seam recovery never inherits a retired stream's high-water mark. Sidecars written after
  the seam bind the incoming identity. *(informative)* The pump's migrate-validated marker (the
  §10.3 step-5 gate) and the switch's config carriage (empty until upgrade records carry a
  config) are host-internal; they are recorded here only to delimit this section's wire scope.

### 12.12 Execution-backend capability, selection, and placement (normative)

The compute@2 runner executes on the execution backend the ADMISSION selected — never an
unconditional CPU runner. Everything below is additive, name-keyed CBOR on the node↔worker
protocol (serde-default decode: pre-extension frames yield the documented defaults),
inventoried for the accumulated WireVersion decision. The architecture's execution-backend
contract (thread/stream discipline, memory-pool headroom, the selection-ladder order, the
platform-correct `DeviceLimits` sources) governs; this section fixes the wire.

- **[XB-1] Backend capability advertisement.** The probe report (`Hardware`) gains
  `backends: [* backend-capability]` (additive; default empty): one record per COMPILED engine
  lane whose runtime probe found a device, plus the always-present CPU record.

  ```cddl
  backend-capability = {
    "backend":      tstr,   ; engine lane slug: "cpu" / "wgpu" / "cuda"
    "class":        tstr,   ; device backend class (the run pre-screen vocabulary):
                            ; "cuda" / "vulkan" / "metal" / "dx12" / "cpu"
    "adapter":      tstr,   ; probed adapter/device name (operator-facing, never branched on)
    "device_index": uint,   ; the device this record describes (placement vocabulary)
    "vram_mb":      uint,   ; dedicated device memory (platform-correct budget source)
    "max_alloc_mb": uint,   ; the per-buffer ceiling (0 = unbounded/unknown) — see [XB-6]
    "shared_mb":    uint,   ; shared/spillover pool (GTT / unified; 0 = none)
    "unified":      bool,   ; shares host DRAM (joint-pool budget math applies)
    "ready":        bool,   ; servable NOW (for CUDA: device AND the two-leg NVRTC gate)
  }
  ```

  A `ready: false` record advertises hardware whose lane cannot yet serve (e.g. a CUDA device
  without its staged, driver-matched NVRTC runtime) — visible to operators, unselectable by
  [XB-2]. The advertisement is the selection input: the ladder consumes exactly these records,
  so what a worker claims and what it selects cannot diverge.
- **[XB-2] Measured selection, no silent fallback.** Selection runs the fixed ladder
  cuda → wgpu → cpu over the advertised records; a rung serves only when its record is `ready`
  and its `class` passes the run's `device_min.backend_class` constraint (empty = no
  constraint). Falling through the ladder is *selection* (the landed rung is recorded and
  revalidated, [XB-3]); once a backend is ADMITTED there is NO fallback: an admitted backend
  that cannot serve is a typed refusal, never a quiet run on another lane. An explicit
  operator lane choice must name a servable, class-allowed lane or the selection refuses typed
  (`cpu` remains the operator's EXPLICIT escape hatch — an explicit CPU selection, not a
  fallback; the former silent downgrade behavior is retired). No servable rung at assess is an
  ineligible verdict with `refusal_code: "BackendUnavailable"` on the `Assessed` surface.
- **[XB-3] The admitted tuple records the selection; join revalidates the device claim.** The
  admitted tuple gains `"backend": tstr` (the selected lane slug; additive, default `""`) and
  `"gpu_index": uint` (the device placement; additive, default `0`). The join-side tuple
  rederivation reruns the identical measured selection and compares these fields exactly like
  the artifact hashes: a device that disappeared or changed between assess and join rederives
  a different selection and the join refuses typed (`admitted_tuple_mismatch` naming
  `"backend"`/`"gpu_index"`) — the node reassesses against the live inventory. A run must
  refuse at join, not fault mid-run, when its assessed device is gone.
- **[XB-4] Device placement.** Placement is node-directed (the `gpu_index` vocabulary:
  `EngineConfig.gpu_index`, delivered to the worker as spawn configuration) and MUST name a
  probed device from the advertisement; naming an absent device is a typed selection refusal.
  Single-device probes place on `0`.
- **[XB-5] Backend unavailability at run start is typed and RECOVERABLE.** A run whose
  admitted backend cannot serve at start (device absent, runtime unstaged, or the process
  device-compute slot occupied) refuses with the typed `BackendUnavailable` error BEFORE the
  run header is journaled; the session classifies it `failed_retryable` (§12.6 [RS-2]) — the
  node reconverges via reassessment. The residual bring-up race (a device dying between the
  start-of-run check and device bring-up on the guest thread) surfaces as a journaled typed
  compute fault (`ComputeFault`), classified the same way. **Deferred device faults at
  fence/readback (driver OOM, host-side allocation rejection) are the CAPACITY class and
  likewise classify `failed_retryable`** — the §15 hardware findings record that a faithful
  driver-OOM diagnostic is not distinguishable from a host-side pool rejection, so the class
  is judged conservatively recoverable and the admitted claim's headroom is the operative
  defense. Hosts MUST NOT infer device health from fence success (§15: fence-visibility is
  closed on the patched CUDA backend but readback remains the authoritative fault surface).
  Module faults stay `failed_terminal`.
- **[XB-6] The per-buffer ceiling is advertised, judged at fleet preflight.** `max_alloc_mb`
  rides the capability record so a run author / the fleet preflight can check the pinned
  model's largest single contiguous tensor against every peer's ceiling (wgpu clamps
  `max_buffer_size` to 2047 MiB on some stacks; Metal's `maxBufferLength` is ≈ ½ RAM). The
  admission funnel does not enforce a per-buffer bound (the claim carries no per-buffer term);
  a breach at runtime surfaces as the [XB-5] capacity class.
- **[XB-7] Process discipline (host-internal, recorded to delimit wire scope).** *(informative
  at the wire; normative for hosts)* At most ONE device-backed compute instance is live per
  worker process (device memory pools never shrink — the peak working set is permanent until
  process exit; a second concurrent device instance is refused as [XB-5] unavailability, and
  real reclamation is the node's worker-respawn discipline). GPU backends are constructed on,
  and driven exclusively from, the per-instance guest thread (stream + pool bookkeeping is
  thread-derived; every compute@2 import is a synchronous host call on that thread, so
  affinity holds by construction). Neither constraint adds wire bytes.

### 12.13 The iroh roster record (normative — lands with the dual-plane transport gate)

A node's iroh transport reachability for one run is published as a **signed roster record**
stored at the registry — the transport analogue of the §12.4 seat lease, under the same
untrusted-storage posture. A record is a canonical-CBOR `roster-record-body` signed by the
node's **certified per-run key** (§12.3), distributed with its `run-key-certificate` beside the
body. The domain tag (registry format `daemon-vhc/<domain>/<semver>`) is
`daemon-vhc/roster-record/1.0.0` — a roster signature is never replayable as a certificate,
frame, seat, or revocation signature, and vice versa.

```
roster-record-body = {
  "domain":        tstr,           ; "daemon-vhc/roster-record/1.0.0"
  "run_id":        bstr .size 32,  ; the genesis hash (§8.1 run_id)
  "role":          tstr,           ; the envelope role label (reader grouping key)
  "epoch":         uint,           ; the certificate's bound epoch
  "incarnation":   uint,           ; §8.1 instance — the freshness key's MAJOR component
  "sender":        bstr .size 32,  ; the certified per-run public key (the signer)
  "module_hash":   bstr .size 32,  ; the certificate's bound module
  "endpoint_id":   bstr .size 32,  ; the iroh transport public key (its own CSPRNG identity, §7.2)
  "direct_addrs":  [* tstr],       ; "ip:port" strings; <= 8 entries, <= 64 chars each
  "relay_url":     tstr / nil,     ; <= 256 chars; nil for direct-only reachability
  "issued_at_ms":  uint,           ; nonzero — the freshness key's MINOR component
}                                  ; distributed as { body: roster-record-body,
                                   ;   certificate: run-key-certificate, sig: bstr .size 64 }
```

- **[ROSTER-1] Registry acceptance is a structural monotonic upsert.** Per `(run, endpoint_id)`
  slot, a registry accepts a publish whose freshness key `(incarnation, issued_at_ms)` is `>=`
  the stored record's (lexicographic; equality is the idempotent republish) and refuses below —
  the stale republish, answered with the stored record as the publisher's re-read. Structural
  checks only: the domain tag, a non-empty role, dialability (at least one direct address or a
  relay URL), the size caps above, slot `run_id`/`endpoint_id` consistency, and a nonzero issue
  stamp. The registry MUST NOT verify signatures or judge authority. A per-run entry cap
  (default 64) gates NEW entry keys only. The shared fold vectors
  (`daemon-vhc-proto/tests/fixtures/roster-vectors.json`) pin these semantics; every conforming
  registry reproduces them bit-for-bit.
- **[ROSTER-2] Peer acceptance.** A peer trusts a fetched record only when: the sender's
  signature over the canonical body verifies; the embedded certificate chains to a
  genesis-trusted base identity (§12.3) and binds exactly the record's
  `(run_id, epoch, role, incarnation, module_hash)` scope with `run_key == sender`. There is no
  wall-clock expiry: staleness is freshness-key precedence, applied per
  `(role, certificate base identity)` group — the durable node key (the per-run key rotates with
  the incarnation; the endpoint id is transport-owned). A reader keeps only the freshest
  verified record per group and never regresses to a lower key, so a withholding registry can
  delay but never roll back discovery.
- **[ROSTER-3] Authoring.** The record is signed by the provisioned per-run key of the
  publishing role-instance; a rejoin (new incarnation) republishes under its fresh key and
  certificate, and a re-addressed node republishes under the same incarnation with a later
  `issued_at_ms`. The advertised addresses and the socket the worker binds MUST agree: the node
  pins the bind address before publishing and delivers it in the plane-selection credentials
  (the additive `bind_addr` of the credentials' iroh half, §12.8-class carriage).
- **[ROSTER-4] Transport surface.** The registry exposes the roster over
  `GET {base}/runs/:id/roster` (the stored snapshot, canonical CBOR `{ "entries": [...] }`,
  storage order — readers verify + reduce themselves) and `PUT {base}/runs/:id/roster` (one
  canonical-CBOR record; an accepted upsert answers 200, a refusal answers 409 carrying
  `{ decision, record }`, a shape error answers 400).

### 12.14 The chunk-addressed det-state contract (normative — the contract and its vectors land first; the ABI ops and host store land with their implementing waves)

Canonical det-lane state (masters, replicated persistents, and — in checkpoints — replica-local
families) becomes host-side, chunk-addressed, hash-verified artifacts under the corpus custody
chain (§12.7), so the guest is a deterministic fold function with memory bounded at
O(chunks in flight) independent of model scale. The contract types, validation rules, and shared
vectors are normative now; the read/write op semantics and the store/retention machinery are
shape-fixed here and land with their implementing waves.

- **[SF-1] Families and layout.** The canonical layout axis is the parameter **registration
  order**; every family is the flat f32-le concatenation of its per-parameter vectors in that
  order — exactly the byte image the resident digest and snapshot sections use. Family names:
  `master` (consensus-canonical; post-ingest, the sealed master of round r IS the round base of
  round r+1 — one artifact, two roles), `replicated:<name>` (consensus-canonical,
  profile-declared), and replica-local families (`ef`, optimizer moments) which appear only in
  checkpoint documents, never in the shared root. Chunking is **per parameter**: a parameter
  never spans a chunk boundary (each parameter independently chunked; last chunk short) — the
  state-plane mirror of [CC-1]'s token rule, making every fold window a (parameter, chunk-range)
  pair.
- **[SF-2] Fold identity, manifest, state root.** A family's artifact identity is the
  domain-separated fold `blake3("daemon-vhc/det-state/1.0.0" ‖ u64le(chunk_size) ‖
  u64le(byte_len) ‖ c_0 ‖ … ‖ c_{n-1})` over its ordered chunk blake3s ([CC-2]'s shared clause).
  The per-round **det-state manifest** (canonical CBOR: format, `run_id`, `round`, the layout
  binding `{params, blake3(canonical numels)}`, `chunk_size`, the consensus-canonical family
  entries `{fold, byte_len, chunk_hashes}`) is derived identically by every peer from its own
  sealed chunks; its blake3 is the **round state root** — an agreement primitive, not a message
  (promotion to an explicit consensus voice is deferred). Validation refuses a
  manifest whose family folds are not the folds of their own chunk lists, whose chunk counts
  contradict per-parameter chunking, or whose family names are neither `master` nor
  `replicated:<name>`.
- **[SF-3] Reads ([SF-R1]/[SF-R2]/[SF-R3] normative as implemented — the trainer wave).** State
  family artifacts are fetched through `data@2::fetch` under [CC-4]'s covering-span verification
  unchanged. Two semantic extensions: a fold the instance itself sealed is registered by
  construction (no `register_chunks`, no grant entry — [SF-R1]; checked AHEAD of the grant gate
  and serviced host-locally from the state store through the ordinary Completion protocol,
  charged against the [CC-5] `data-read-budget`; an evicted root falls back to the grant gate and
  refuses typed); externally-sourced roots (genesis init pin, restore descriptors) are granted +
  registered like corpus shards ([SF-R2], [CC-3] posture). Fold windows combining state ranges
  with device-export buffers use ranged `read_into` ([SF-R3]). Implementation note ([SF-1]
  consequence): per-parameter chunking is NOT a uniform grid — interior short chunks exist at
  parameter tails — so the host store records each emitted chunk's length and resolves ranges by
  walking actual offsets; the corpus `covering_span` arithmetic applies only to uniform-grid
  registered maps. **[SF-R2] registration descriptor (ruled + implemented).** An externally
  sourced root registers through a `data@2` **det-state chunk map** at **minor 3**
  (`register_state_chunks`): per-chunk `(hash, len)` pairs the guest derives from the layout
  (numels + `chunk_size`), keyed under the `daemon-vhc/det-state/1.0.0` fold domain. The host
  builds a length-aware registered map — the symmetric twin of the self-sealed store's `(hash,
  len)` model — and a det-state-aware covering-span verifier that walks the actual per-chunk
  offsets; `FamilyRef` and the `DetStateManifest` wire shapes are unchanged. The verification
  integrity is the deciding property: **the fold pins the ordered chunk hashes; the lengths are
  only framing hints, so a lying descriptor fails blake3 re-verification rather than corrupting
  silently** — root minimality and digest continuity are preserved with no new trust surface.
  Rejected alternatives (lengths in the manifest; a layout-aware host) are declined — the second
  because host layout-agnosticism is a deliberate design posture. This descriptor path is binding
  on the restore wave: restore roots enter the same way.
- **[SF-4] Writes (normative as implemented — the host state store wave; shapes as reserved).**
  Three `vhc@2` imports at **minor 3**: `state_open(tag_ptr, tag_len, byte_len) -> stream`
  (counter-deterministic top-bit stream ids), `state_emit(stream, ptr, len) -> ordinal`
  (host copies, blake3-hashes, stores content-addressed; a misframed emit — empty, larger than
  the run-pinned `chunk_size`, or past the declared `byte_len` — traps the typed
  `StateMisframedEmit`; framing is deliberately coarse, per the ratified ruling: per-parameter
  tail alignment is a fold-identity concern the windowed≡resident parity suites and the init
  `expected_root` catch, never a host trap), `state_seal(stream, out_ptr) -> u32` (the
  domain-separated family fold over the accumulated chunk hashes written back; an incomplete
  seal traps the typed `StateIncompleteSeal` and leaves the stream open for completion+retry).
  Determinism classes: open/emit are `dc` (re-executed at replay over reproduced guest memory
  into a replay-side state chunk store); seal is `nr` (the journal-record-only tag-2 **kind 6**,
  `src` = the sealed stream id — the recorded 32-byte fold is the O(1) divergence cross-check
  at replay). An opened-but-unsealed stream is never durable: torn folds force-reclaim at
  instance teardown, and the store is instance-scoped, so a crash mid-fold leaves nothing a
  restart or a fetch can observe.
- **[SF-5] The genesis state contract (normative as implemented — the trainer wave).** The
  contract is now shipping-consumed: the trainer guest derives its init from `state_contract`
  (seed expansion cross-checked against `expected_root`, or artifact-form fetch) and the bulk
  inline `GuestCfg.init` is deleted. Because the contract becomes wire-visible here (dormant
  until now), **`VhcProtoVersion` bumps 1 → 2** with this wave (the §10.3 wire-change trigger);
  the app↔node `WireVersion` is unchanged (no DTO change). The genesis envelope gains an additive
  `state_contract` key: `{chunk_size, init}` with `init` either seed-derived
  `{seed: bstr32, dist: uint, expected_root: bstr32}` (deterministic chunk-wise expansion,
  sealed fold cross-checked against the pin — a mismatch is a typed init failure) or a
  content-addressed `{manifest: bstr32}` (a det-state manifest that MUST also name a fetchable
  `[artifacts]` entry; the warm-start and golden-continuity form). The bulk inline
  `GuestCfg.init` is deleted by the trainer wave. **Authoring validation rules (normative,
  vector-pinned):** the compression profile's `chunk` MUST divide every parameter's numel (for
  the frozen ceremony geometry the 1536-wide norm parameters make `chunk | 1536` binding — the
  profile default 4096 is a refusal); `chunk_size` MUST be a non-zero integer multiple of the
  profile chunk's byte width (`chunk × 4`), derived nearest ~4 MiB; and the remote checkpoint
  cadence plus one publisher-churn slot MUST fit inside `payload_retention_rounds`
  (`0` = unbounded retention; a configuration that could strand a rejoiner past retention is
  refused at authoring).
- **[SF-6] Checkpoint document v2 (normative as implemented — the checkpoint/restore wave).**
  Sections are inline (`[name, bytes]` — small sections like the 8-byte round watermark) or
  by-reference (`[name, family-ref]` with `family-ref = {fold, byte_len, chunk_size,
  chunk_hashes}` — already-sealed family artifacts, zero additional bytes moved); the document is
  the 2-element array `[manifest_bytes, [ckpt-doc-section…]]` under one shared codec
  (`daemon_vhc_proto::det_state::{encode,decode}_checkpoint_doc`). A live checkpoint of the
  flagship trainer references `master`/`ef`/`adamw_m`/`adamw_v` by fold (the families are already
  sealed) plus the inline `round`; the structural recognizer verifies a by-ref section by the
  ref's own fold consistency (its fold IS the fold of its listed chunks) and the host store
  holding those chunks, never by re-hashing inline bytes. **Restore is streaming rehydration**:
  the migration descriptor's `migration-section` gains a by-reference alternative
  (`{name, family}`) beside the inline `{name, staging_id}`; `da_migrate` records the family
  refs (no bulk read — §6.6 unchanged), and `da_run` registers each fold ([SF-R2]) and streams
  its windows on demand with bounded in-flight refill (never exceeding `max_outstanding_ops`),
  materializing no whole family in guest memory. The DRAIN path carries the same by-ref sections
  reconstructed host-side from the draining instance's own state store (the guest names the
  sealed fold; the host fills the geometry). **Cadence/publisher policy:** the local
  cadence gates the checkpoint boundary (the folds already exist — a local checkpoint is pointer
  bookkeeping); the remote cadence + a ONE-per-slot deterministic publisher election
  (checkpointer-salted, slot-rotated) gate the upload, so a replicated group uploads once per
  slot; a dead slot's publisher is covered by the next slot's rotation (the one-slot slack). The
  publisher uploads the by-ref document + its family chunks content-addressed to the payload
  plane — idempotent, so chunks unchanged since a prior slot upload nothing (skip-on-present).
  The [SF-5] cadence↔retention bound governs the remote cadence, enforced at genesis authoring.
  Referenced folds are pinned out of retention eviction while their checkpoint is the freshest of
  its (role, kind) slot ([SF-7]). Because the descriptor + document are `VhcProtoVersion`-2
  shapes (already bumped by [SF-5]), this wave adds no wire-version movement; the app↔node
  `WireVersion` is unchanged.
  **In-process live-module-switch carriage (normative as implemented).** A live module switch
  (§10.3) is an in-process migrate on ONE node, not a content-plane restore. The switch
  transaction carries the drain snapshot's sealed families directly from the draining instance's
  state store into the SUCCESSOR instance's state store (lifted while the draining pump is still
  alive, before it is retired — so a crash mid-switch never leaves the chunks existing nowhere),
  and the successor's run config inherits the run-pinned `state_chunk_size` (the genesis state
  contract is not re-read at re-admission). The successor therefore serves the drain snapshot's
  folds **self-sealed** ([SF-R1], host-local): the local switch publishes nothing to the payload
  plane and fetches nothing from it — the host retains custody of canonical state across the
  fence, preserving the zero-bytes-moved intent. This is the sole distinction from a rejoiner's
  late-join restore, which streams the *published* checkpoint from the content plane ([SF-R2]);
  the carried folds are retention-pinned so a concurrent seal cannot evict them out from under the
  successor's streaming restore walk. Local switch ≠ content-plane restore.
- **[SF-7] Grants and retention (normative as implemented — the host state store wave).** The
  grants vocabulary gains `state-write-budget` (raw bytes, token bucket + per-emit ceiling,
  enforced at `state_emit`; breach traps `GrantViolation` — writes are guest-driven, so
  attributable; the token bucket runs on logical pump time and is live-only, the epoch-watchdog
  posture — replay is not the budget gate), `state-store-bytes` (live retained bytes across
  sealed families, enforced at `state_seal` AFTER retention eviction; a seal that would still
  exceed it is refused typed and **rolled back** — nothing durable remains), `state-streams-max`
  (concurrent open streams, enforced at `state_open`), and `state_retain_roots` (sealed roots
  retained per family beyond the checkpoint-pinned set; default 2 — the round base and the
  freshly sealed round; eviction is per-artifact, oldest-seal first, chunks refcounted so
  cross-round identical chunks store once), all derived tighten-only like every §2.6 bound
  (carried in the `vhc@2` world grant's generic bounds map until a lane declares state
  ceilings). State reads ride the existing [CC-5] `data-read-budget`.
- **[SF-8] The streaming digest carry and the slice-decomposition schedule (normative now,
  vector-pinned).** Because full coverage is sequential, the exact round state digest
  (`digest_state(seed, 64, u32::MAX, state)`) is computable as a **carry**: the seeded
  streaming hasher + the absolute byte offset, injecting the 8-byte LE block-index frame at
  each 64-byte boundary independent of update splits — bit-for-bit equivalent for ANY chunking
  of the state (digest values, coverage, and formula unchanged; existing pinned digests are
  the refactor's parity evidence). Full coverage holds while the block count fits `u32::MAX`
  (< 256 GiB at block 64). The streamed ingest/make_update walks are **completion-driven
  multi-slice state machines** under a pinned schedule: per-parameter window enumeration
  (window ordinal ≡ family chunk ordinal), folds ascending-and-contiguous regardless of
  completion arrival order, reads bounded by the in-flight window, the seal exactly once in
  the slice folding the last window — so per-slice work is bounded by construction and the
  walk's operation order is the resident order, window-sliced (the windowed ≡ resident parity
  obligation of the fold-engine wave). The degenerate single-window geometry (the 64-dim
  acceptance tier) is the same code path.

---

## 13. Conformance surface

A conforming implementation MUST pass (mapped to the refactor's CI tiers, §10):

- **Selection + negotiation**: a v2-major module is refused by the v1 path and vice versa with
  typed refusals, not traps; import/declaration mismatch → `AbiDeclarationMismatch`; minor-too-new
  refused; minor-lower admitted; per-world minor validation against actual static imports; bridge
  modules select major 2.
- **Two-instance model**: a module calling any capability import from `da_manifest`/`da_claim`
  traps `ClaimCapabilityDenied` and is refused; the assessment instance is provably discarded (no
  state carries into the run instance); `da_init` receives byte-identical admitted config/grants
  (tag-11 hash check).
- **The closed subset**: a module using only `{next_event, publish, set_timer, cancel_timer,
  read_back, now}` runs with zero host changes; a non-round timer-driven averager is the first
  expressiveness proof (refactor §5 A2 acceptance).
- **Bridge parity**: TinyLlama on `BarrierRound` over the `tabi@1` bridge reproduces v1 det-lane
  state digests across `cpu`/`ndarray`/`wgpu`/`cuda` (refactor §5 Phase A acceptance) — the
  control-inversion-didn't-change-math evidence.
- **`next_event` storage protocol**: `NeedCapacity` round-trip (small buffer → exact required
  length → re-call succeeds; event not consumed; budgets not reset; replay unaffected).
- **Stop vs Quiesce**: imports after `Stop` trap `PhaseViolation`; a drain completes within the
  deadline with `QuiesceReady`; deadline expiry produces `QuiesceDeadlineExceeded` via epoch
  interruption with guest-thread teardown.
- **Channels**: publish on an undeclared / rx-only channel traps `GrantViolation`; oversize
  payload traps `PayloadOverflow`; channel-scoped gap detection does not fire on advisory drops.
- **Journal + input replay (bit-exact)**: a recorded sim run replays bit-exact decisions including
  handle values and timer IDs; the coordinator oracle passes unchanged over the segmented
  substrate (refactor §5 A1); **crash-recovery test**: kill between write and barrier → truncate →
  seq counter never reused → replay of the truncated journal passes; **missing-payload test**:
  replay without a referenced payload reports `ReplayMissingPayload`, never a pass; **terminal
  injection test**: a recorded epoch trap replays as the recorded fault at the recorded ordinal.
- **The det-state contract vectors (§12.14)**: the shared chunk-geometry vectors (ceremony-
  geometry profile-chunk refusals incl. the 4096 default at the 1536-wide norms; state-chunk-size
  multiples + derivation; cadence↔retention bounds), the digest-carry equivalence vectors
  (pinned golden digests reproduced bit-for-bit one-shot, per fold window, per byte, and at
  unaligned strides — parameter tails, multi-chunk parameters, window ≡ block framing, partial
  final blocks), and the fold-walk schedule vectors (pinned window enumeration; fold order
  invariant under arbitrary completion arrival permutations; bounded in-flight reads; the seal
  exactly once). **Landed with the host state store**: state-op conformance (the write
  vocabulary end-to-end with the sealed fold reproducing the proto family fold bit-exactly;
  framing + grant traps typed; the [SF-R1] self-sealed fetch; degenerate single-window store
  geometry; torn folds never durable across teardown and restart) and the state replay suites
  (emit re-execution into the replay-side store; the kind-6 seal cross-check tripping on a
  divergent recording; a torn-fold journal replaying cleanly; self-sealed fetches materializing
  with no payload archive). **Landed with the SDK fold engine**: window ≡ resident fold parity
  for the flagship profile — the streamed ingest/make_update walks reproduce the resident
  implementation bit-for-bit (emitted masters, payload sections, error-feedback state, and the
  digest carry finalizing to the resident digest formula) across window geometries (degenerate
  single-window, one-chunk windows, short parameter tails, a ceremony-shaped scaled layout),
  in-flight bounds, and completion arrival permutations; the profile walks execute the shared
  fold-walk schedule vectors directly (enumeration + fold/issue/seal order); mis-geometry and
  walk-protocol violations refuse typed. Later waves add: fold parity for the remaining
  profiles (their resident APIs stay as the oracle until the resident-path deletion wave),
  real-guest state-op conformance through the SDK (the trainer wave, which owns guest re-pins),
  and the 64-dim whole-run degenerate-geometry assertion.
  **Resident-path deletion adjudication (settled — do not re-litigate from stale delete-intent):**
  the resident-path deletion wave ADJUDICATED RETENTION of the resident profile
  implementations (`SparseLoco`/`DiLoCo`/`Demo`) as the pinned-golden reference AND the streaming
  window-parity oracle, dev/test-quarantined (the production trainer guest links only the
  streaming fold engine + the profile config; no production/guest path links the resident
  profiles). Rationale: the window ≡ resident parity proof is NOT self-standing — three surfaces
  (the parity suite, the schedule-vector suite, and the trainer-golden capture harness) recompute
  the resident oracle live, and a live oracle is strictly stronger evidence than a frozen capture
  of itself; and the non-flagship profiles have no windowed counterpart and no production
  consumer, so windowing them would be speculative surface. Only genuinely-dead resident surfaces
  the streaming path superseded (e.g. the resident error-feedback checkpoint-restore method,
  replaced by streaming rehydration) were deleted. This is the ratified retain-or-window choice
  resolved to RETAIN with documented rationale.
- **Claim path**: over-claim rejected against owner policy; under-claim traps attributably at the
  cap; `ClaimInconsistent` on a nondeterministic claim; all tier-1.
- **Budgets/watchdog**: per-slice fuel/op reset on `Delivered` only; a spinning slice traps
  `BudgetEpoch`; a parked guest is never epoch-killed; bridge calls charge the op budget.
- **Fail-closed decoding**: an unknown event tag / status / enum value traps — the
  advisory-no-op behavior is asserted absent.
- **Timers**: cancel-before-fire suppresses delivery (status 0); cancel-after-fire returns 1; IDs
  never reused within an instance.
- **Resource classes**: slice handles trap `StaleHandle` across a slice boundary; instance handles
  trap `StaleHandle` after restart and are re-acquirable; generation-wrap slot retirement
  asserted; generations replay-deterministic across a trap-restart (tag-13 seeded).
- **Journal grammar validation (tier-1)**: the §8.3 CDDL grammar validates as-is under cddl-cat,
  and every record of every conformance-run journal validates against it; a grammar-rejected
  record fails the gate.
- **Bridge replay registry coverage**: the §2.7 registry covers `TABI_IMPORTS` exactly (no import
  unclassified, none classified twice); every nr-class import's result is asserted journaled;
  a dc/dd handle value reproduced by bookkeeping alone matches the recorded guest behavior.
- **Cell-5 envelope fixture** (§9.3, all three provisos): (a) old `FrozenEnvelope::open` accepts +
  verifies new raw bytes carrying `device_min`; (b) original bytes/hash preserved end-to-end;
  (c) no decode→re-freeze path exists (asserted by construction/inspection test) — REQUIRED before
  Cell 5 ships; on failure Cell 5 is refused until D0.
- **Grants pipeline**: the derived grants document round-trips byte-identically from admission
  through `da_claim`, the tag-0 header, and `da_init` (tag-11 hashes); assess→join hash-pin
  mismatch restarts admission; a stale-policy join is asserted refused.
- **Migration scaffolding**: snapshot round-trip in sim (`stage_state` → `snapshot_state` (with a
  rejected-then-retried submission) → host verify → instantiate + tag-13 **before** `da_init` →
  `da_migrate(descriptor)` → `read_back(state-section)` during `da_migrate` → `Ready`);
  `MigrateUnsupported` surfaces as an admission refusal.
- **Signing**: every A2 outbound frame carries the full §12.1 envelope; equivocation-evidence
  comparison over the complete scope tuple is exercised; a frame missing any scope field is
  rejected at verification.
- **Encoding**: canonical-CBOR determinism (two encodings byte-identical); additive minor growth
  (older reader ignores trailing fields; above-minor delivery asserted absent).

Conformance runs against the **production wasm blob** in `host/daemon-vhc-testkit` (wasmtime +
simulated capability providers), never against SDK-native sim — the split the architecture mandates
(§6) and the dependency rules enforce.

---

## 14. Summary of retained-vs-new (informative)

| Mechanism | Source | v2 disposition |
|---|---|---|
| `da_abi` packed `(major<<16)|minor` | v1 `runtime.rs`/`lib.rs` | retained; cross-check against import-derived candidate (§1.3) |
| `(ptr<<32)|len` CBOR return, `da_alloc`/`da_free` | v1 `rt.rs` | retained for CBOR-returning exports; never used inside imports (§2.4) |
| `tabi@1` import vocabulary (66, frozen) | v1 `abi.rs` | retained as the transitional compute bridge under major 2, retiring at Phase C (§2.5) |
| Handle encoding (kind/gen/index), `StepArena` generations | v1 `handle.rs` | retained; three-tier resource classes, journaled generation seeds, wrap-retirement (§7) |
| Trap taxonomy (21 codes, typed, subprocess survives) | v1 `trap.rs` | retained + 6 v2 codes; `MigrateUnsupported` reclassified as refusal (§7.6, §1.5) |
| Canonical CBOR + ed25519 signing | `daemon-vhc-proto` | retained; channel-scoped durable seqs + the domain-separated frame envelope from A2 (§12.1) |
| `MessageLog`/`RunCapture` (in-memory, clocks captured) | `daemon-swarm-observe` | generalized → segmented crash-safe journal with barriers, evidence, sidecars (§8) |
| Phase-legality table | v1 `phase.rs` | **retired for v2**; three temporal rules + SDK type-state (§6.6) |
| `MetaReport` host probe admission | v1 `meta.rs` | **replaced** by guest `claim()` in the assessment instance (§9) |
| Host-sequenced 5-phase lifecycle | v1 | **replaced** by `da_init` + `da_run` + `next_event` (§3–§4) |
| `Command`/`Event` node↔worker protocol | `daemon-swarm-run` | evolves additively (role-instance ids); orthogonal to this ABI |

---

## 15. Reserved for later phases (index)

- **`compute@2` tensor surface** — command queue + `fence`, device `read_back`, autodiff split,
  det reclassification. Ratified as the **Phase C** entry gate; reserved: `Fence` event (tag 5);
  handle kinds 1–5 carry over from the bridge. The bridge (§2.5) retires here.

  **Ratified direction (2026-07-15, post-spike; decisions D8).** The Phase-C entry gate — the
  Burn-over-`HostBackend` prototype spike (findings archived at
  `docs/research/burn-backend-findings.md` in the node tree; tier-1 bit-exact) — passed, and the human ratified the
  direction the Phase-C compute section MUST elaborate. The surface stays **reserved** (no wire
  bytes, no linked imports before Phase C); the following is the ratified frame, superseding this
  bullet's former "Burn-shaped codegen" phrasing:

  1. The `compute@` payload is **CBOR-encoded `burn_ir::OperationIr` at the pinned Burn version
     per ABI major** (`burn = 0.21.0` in-tree). The IR's variant set/discriminants/fields are
     Burn-version-specific: **a Burn version bump is an ABI event** (variant insertion is a
     compute-major). Host-side dispatch reuses the `burn-router` runner; the governance point is
     the pinned IR schema + a conformance suite, never a hand-curated op list.
  2. **Handles:** one opaque `u64` `TensorId` per tensor (all ranks, all kinds; rank/dtype are
     runtime data in the IR), instance-class, **guest reference-counted**, released via
     `OperationIr::Drop`; operand `TensorStatus` (in-place hint) is preserved on the wire; N
     output handles per op are allowed.
  3. **Metadata is guest-authoritative:** shape/dtype/rank live guest-side (`RouterTensor`);
     burn-ir's shape-inference is part of the pinned contract — the host MUST produce outputs of
     the guest-computed shape; no host metadata import exists on the hot path.
  4. **Errors are deferred:** enqueue is infallible; errors surface at fence/`read_back` only.
     **A stale/unknown handle is a typed `StaleHandle`/`InvalidHandle` trap (§7.6)** — the mapping
     from Burn's `ExecutionError` and runner panics into the §7.5 `comp-error` codes / §7.6 slugs
     is a Phase-C obligation (the upstream runner panics today; that panic MUST never surface as
     a host crash).
  5. **RESERVED, refuse cleanly until specified:** quantization/`QFloat`
     (`Quantize`/`Dequantize` do not lower today) and `OperationIr::Custom` (custom/fused kernels
     stay in the host-side custom-op registry, architecture §3.2); `Distributed` is out of scope.
     The rank-erased-primitive property of Burn is a hard ABI floor.

  No autodiff ABI surface is needed (the tape is guest-side over the same enqueue/handle
  primitives; proven with zero intermediate readbacks) — `backward@1`/`grad@1` retire under
  `compute@2` with the bridge. Bulk `TensorData` upload/readback rides `BufferHandle` (§7.4),
  never inline in the op-stream. Phase-C caveats carried from the spike: CUDA/wgpu deferred-error
  *timing* is unexercised (ndarray-CPU only) and MUST be exercised in C.

  **Hardware conformance note (2026-07-16; RTX 4090, vhc-integration `9a7185c3`, cuda-gated
  evidence test `compute_cuda.rs` landed at `d856538e`; corrected same day — audit, see §0.5
  a3).** A host-side allocation rejection exercised rule 4's deferred-error path: a 2×-VRAM
  (50.5 GB) single-buffer `Full` request was refused by cubecl's memory pools
  (`IoError::BufferTooBig` — no pool's `max_alloc_size` accepts it) before any `cuMemAlloc`;
  the driver was never engaged and no `CUDA_ERROR_OUT_OF_MEMORY` occurred. Rule 4's readback
  half held for this case: enqueue stayed infallible and the fault surfaced typed at
  `read_back` of the affected handle (`ComputeError::Device`, trap twin `ComputeFault`; no
  panic/abort escaped) — with the qualification that readback carries a generic
  runner-panicked fault for that handle, not a faithful allocation diagnostic. The fault is
  scoped to the one unbacked handle: the same runner (and a fresh runner) keeps serving new
  work. But `fence()` returned Ok after the fault. Root cause: `cubecl-cuda`'s alloc path
  (`initialize_memory`) panics on the fallible reserve instead of pushing the error into the
  stream's error queue the way the `launch` path does; the panic is caught per-task and
  dropped, so no error state exists for `RunnerClient::sync` to drain (the sync plumbing would
  surface a `ServerError` if one existed). **Fence-visibility of deferred device faults is
  REQUIRED by this spec but currently unmet on the CUDA backend (readback-visible only).**
  Conforming hosts MUST NOT treat a successful fence as evidence of device health until the
  cubecl fix lands (make the alloc path record errors like the launch path so fence/sync can
  drain them; tracked post-Phase-E follow-up); guests SHOULD treat readback as the
  authoritative fault surface in the interim, with the generic-fault qualification above.
  Residual (broader than first recorded): genuine driver-reported `CUDA_ERROR_OUT_OF_MEMORY`
  and sticky-error semantics (e.g. illegal address, unreachable through the pinned op set)
  remain unvalidated — the spike's escalated deferred-error gap is still open; only this
  host-side pool-cap rejection case is closed.

  **Hardware conformance update (2026-07-17; RTX 4090, driver 580.159.04, ~24080 MiB free; burn
  0.21 / cubecl 0.10.0; rustc 1.96.0; branch `vhc/cuda-fence-visibility`, cuda-gated evidence
  test `compute_cuda.rs`). The fence-visibility gap is now CLOSED on the CUDA backend, and a
  genuine driver-reported OOM has been exercised.** Root cause of the gap (recorded in a3) was
  confirmed and fixed: `cubecl-cuda`'s `CudaServer::initialize_memory` did
  `command.reserve(size).unwrap()`, panicking on a fallible reservation on the device-stream
  thread; cubecl catches that per-task panic and drops it, so no error reached the stream error
  queue and `RunnerClient::sync` (the `fence()`) returned `Ok`. A vendored, single-crate patch of
  `cubecl-cuda 0.10.0` (wired via `[patch.crates-io]`) makes the alloc path record the failed
  reservation on the stream error queue exactly as the `launch`/`write` paths already do
  (`command.error(err.into())`), so `sync`/`fence` drains it as `ServerError::ServerUnhealthy` →
  the typed `ComputeError::Device` (trap twin `ComputeFault`). It is a one-idiom change (an
  `.unwrap()` replaced by the crate's own error-recording path) and is a clean upstream-PR
  candidate.

  Validated on the 4090 (all cuda-gated, `--test-threads=1`):
  - **Fence-visibility (host-side pool-cap rejection).** A single ~50.5 GB (`>2× VRAM`) `Full`
    buffer, refused by the pool cap before any `cuMemAlloc` (driver never engaged), now surfaces
    at the fence typed (`ComputeError::Device`/`ComputeFault`) — previously `Ok` at the fence,
    readback-visible only.
  - **Genuine driver `CUDA_ERROR_OUT_OF_MEMORY`.** Allocations that are each individually
    pool-acceptable (2 GiB requests, well under the ~VRAM pool cap) but sum past free VRAM (19 ×
    2048 MiB = 38912 MiB vs 24080 MiB) force cubecl's pool to grow in ~6.3 GB pages until the
    driver's own `cuMemAlloc` fails — the driver IS engaged (unlike the prior single 2×-VRAM
    buffer). Enqueue stays infallible; the fault is **fence-visible AND readback-visible**, typed;
    the host and CUDA context survive (the same runner and a fresh runner both keep serving). This
    is the reachable **non-sticky** driver-fault class.
  - **A/B control.** Rebuilt against pristine (unpatched) `cubecl-cuda 0.10.0`, the two
    fence-visibility cases fail because `fence()` returns `Ok(())` after the fault (the raw
    `initialize_memory` panic is caught and dropped); rebuilt with the patch, all cases pass —
    isolating the fix as the cause.

  **Residual still open (honest):** genuinely **sticky** CUDA error semantics (an illegal-address
  kernel that poisons the context so every subsequent op fails until context recreation) remain
  unvalidated because they are **unreachable through the pinned op set** — every servable
  `burn_ir::OperationIr` variant is a bounds-checked tensor op, and the only escape to a
  hand-written kernel, `OperationIr::Custom`, is refused pre-dispatch (`CustomOpUnsupported`,
  RESERVED). The driver-OOM readback surfaces a generic "runner panicked (unknown handle)"
  `Device` fault, not a faithful allocation diagnostic; and the GPU storage layer maps a real
  driver OOM to the same `IoError::BufferTooBig` variant as the host-side pool-cap case, so the
  drained fault does not by itself distinguish driver-engaged from host-side (established here via
  the stimulus design and the ~6.3 GB page sizes, not the error variant). A faithful diagnostic
  and the sticky class would need, respectively, the driver error class threaded through the
  storage layer and a raw/custom-kernel escape the host deliberately refuses.
- **Async completion protocol** — `OpId` (kind 10), `Completion` (tag 6), `vhc@2::cancel`,
  credit-based streams. **Phase B**; result encoding fixed in §7.5.
- **Buffer layer** — `BufferHandle` (kind 8), `read_into`/`create_from`. **Phase B** (§7.4).
- **`net@2`/`data@2` beyond Phase A** — payload put/get, gossip, streams, artifact fetch. **Phase
  B**; additional channels enter the channel table.
- **Envelope-declared channel tables** — **D0** (§6.2); ABI surface unchanged.
- **Certified per-run key chains** — **shipped and extended by the certified-identity work**: the
  certificate binds the per-run key to the full execution identity `(run_id, epoch, role,
  instance, module_hash)`, with signed revocation records and incarnation supersession (§12.3).
  The frame envelope and seq semantics remain final from A2 and are not altered (§12.1, §12.2).
  Broader `Authority`/`Committed<T>` record semantics continue to layer on top.
- **Coordinator seat lease** — **shipped (node-side contract + registry CAS semantics)**: the
  Authority-signed fenced lease over a run's coordinator role (§12.4) — token≡incarnation
  fencing, untrusted-storage registry CAS with the tombstone floor, peer-side verification
  chaining into §12.3's certificates and supersession. The cloud registry implements the same
  frozen surface and shared CAS vectors when its coordinator surface next lands.
- **Record archive** — builds on §8.2's chained segments + §8.6 evidence records. **Phase D**.
- **Upgrade transaction** — the §10.3 sequence, implemented **Phase E** over the §10.2 scaffolding.

---

## 16. Resolution log (ratification review)

Draft 1's open questions were resolved by the first adopted review; Draft 2's nine blockers were
resolved by the second adopted review. Both are folded into the normative body; this section is the
audit trail.

**Second review (Draft 2 → Draft 3):**

| # | Resolution | Where |
|---|---|---|
| R2-1 | Canonical top-level grants document: CDDL, lane ∩ envelope ∩ owner derivation, Phase-A form, assess→join hash-pinning; **no signed assessment token** (assess and join share one node client's trust domain — a token has no threat model there) | §2.6, §9.4 |
| R2-2 | §9.6 is the single canonical `ParticipationLane` schema (decisions D7 references it, restated schema deleted); raw bytes/bps units only; `bridge_allowed` + `replay_window` added; claim-tier → admission/arbitration mapping (all three tiers reserved in the ledger — D6 amended); `Budget` event host-fixed, outside event-caps | §9.6, §4.3 |
| R2-3 | Migration bootstrap made workable: `stage_state` import; migration descriptor carries restore staging IDs directly; `read_back(state-section)` legal during `da_migrate`; tag-13 journaled at instantiation before init/migrate; "one successful submission, rejected attempts may retry" | §10.2, §6.4, §6.6, §10.3 |
| R2-4 | §8.3 is one machine-valid tagged-union CDDL grammar, cddl-cat-validatable, tier-1-enforced; seal hash excludes the seal record; `NeedCapacity` is a mandatory-retry protocol rule (proceeding without retry traps), hence recordless by design | §8.3, §8.2, §4.1, §6.4 |
| R2-5 | The minimum domain-separated `frame-envelope` (exactly the scope-tuple fields) lands at **A2**; D1 adds certified keys + `Authority` and MUST NOT touch the envelope fields — the only reading consistent with OQ-15 | §12.1, §6.2 |
| R2-6 | Exhaustive bridge replay registry: all 66 `tabi@1` imports classified dc/dd/nr/se; every control-flow-influencing result det-lane or recorded-verbatim — the replay theorem | §2.7 |
| R2-7 | Sidecar AEAD profile: XChaCha20-Poly1305, one node-local key per journal (secret storage, never journaled), nonce = LE64(ord)‖LE64(instantiation)‖LE64(0), header CDDL as AAD, retention/access hooks; logical time: run-join epoch, sampled once per delivered event before delivery, slice-constant `now()`, journaled `at`, restart high-water clamp, replay from recorded readings only | §8.5, §6.5 |
| R2-8 | Execution identity frozen as `(run_id, epoch, role, instance, module_hash)`; applied to segment/run headers, sidecar ownership, the signed envelope + domain separation, sequence scopes, and the admission machine; companion docs aligned (decisions D1, refactor §9, architecture §5.1) | §8.1, §8.2, §8.5, §12, §9.4 |
| R2-9 | Cell 5 ratified **conditionally** on the three-proviso fixture (old-reader open, bytes/hash end-to-end, no decode→re-freeze — the silent-field-discard trap called out); `device-minimums` CDDL defined; on fixture failure Cell 5 refused until D0 | §9.3, §13; decisions D3 |

**First review (Draft 1 → Draft 2):**

| OQ | Resolution | Where |
|---|---|---|
| OQ-1 | Broad internal category (`AbiMismatch`) + split exposed codes | §1.5 |
| OQ-2 | Loop mechanics in `vhc@2`; routing in `net@2`; clock/timers/telemetry in `sys@2` | §2.2 |
| OQ-3 | Guest-provided buffer with `NeedCapacity` (no host allocation, no reentrancy) | §4.1, §2.4 |
| OQ-4 | `budget-report` fully defined (wire-gap mandate; resolved editorially) | §4.2 |
| OQ-5 | Outcome vocabulary stays `{Ok, Left, QuiesceReady}`; ≥16 module-defined ≙ `Left`; 3–15 reserved (resolved editorially) | §4.5 |
| OQ-6 | Bounds declared per channel (default table now, envelope from D0), constrained by lane ceilings | §4.7, §6.2, §9.6 |
| OQ-7 | `read_back` charged in-slice, normative (resolved editorially; per-window budgets deferred to the Phase C compute ratification) | §5.5 |
| OQ-8 | Replay injects the recorded terminal fault at the recorded ordinal; wall-clock never re-armed | §5.6, §8.7 |
| OQ-9 | Synchronous durable-spool acceptance returning the seq; network outcome later via completion (Phase B) | §6.2, §8.4 |
| OQ-10 | Typed u64 timer ID scoped to one instance; not a resource handle; handle kind 11 unassigned | §6.3, §7.2 |
| OQ-11 | Owner-configured quiesce deadline bounded by the lane's `quiesce_deadline_max_ms` | §4.4, §9.6 |
| OQ-12 | Per-segment BLAKE3 chain + per-record CRC32C | §8.2 |
| OQ-13 | Additive envelope field accepted, REQUIRED old-reader/new-envelope fixture test; defer-to-D0 fallback recorded by A2 if the fixture cannot pass *(refined into the three-proviso conditional by R2-9)* | §9.3, §13 |
| OQ-14 | One dedicated guest OS thread per live role-instance | §11.1 |
| OQ-15 | Final durable domain-separated channel-scoped sequence semantics from Phase A; no interim meaning *(schema landing fixed at A2 by R2-5)* | §6.2, §12 |

Remaining deliberately-deferred items (bracketed by the architecture, not open questions of this
document):

- **The `compute@2` lowering rules** (pinned Burn version per major, associated types, rank/dtype
  genericity, tensor-metadata ownership, handle lifetime, error model, autodiff guest/host split)
  are a whole ratified section owned by the Phase C entry gate; this document reserves the surface
  (§15) and the bridge (§2.5) but does not decide it — the Burn-over-`HostBackend` prototype must
  validate it first (refactor §7 entry criteria). *Update (2026-07-15): the prototype spike has
  validated all six rules and its reframing is ratified (decisions D8); the §15 reserve now
  records the ratified direction the Phase-C section must elaborate. The full normative compute
  section remains Phase-C work.*
- **Confinement vs information-flow** (architecture §10): the ABI bounds *where* bytes go and how
  fast, not *what they encode*. Journals, metrics, logs, and sidecars are egress channels under
  the same grants (§8.5 additionally encrypts sidecars at rest). This is a stated non-goal;
  ratifiers signing off on `publish`/`emit_metric`/`log` as egress should note it explicitly.

---

## 17. ABI major 2, minor 5 — the certification minor (NORMATIVE)

*Ratified 2026-07-26 (normative amendments A1 and A5, and the implementation rulings that follow
them). This section is normative where it and the body of this document differ. It supersedes §9's
`claim()` surface, extends §6.5's `log` bounds, and replaces §7.6's trap-context enumeration.*

Minor 5 carries **three coordinated changes** in one bump. They share the bump because each changes a
declaration, an export or a linked contract in a production module, so landing them separately would
invalidate a certification candidate's pins three times over. They are not one semantic reason.

### 17.1 The declaration ladder

| Constant | Value | Meaning |
|---|---|---|
| `DA_ABI_MAJOR_V2` | `2` | the major this document fixes |
| `DA_ABI_MINOR_V2` | `5` | the highest minor this host implements |
| `CERTIFICATION_MINOR_V2` | `5` | the minor at which the surface below applies |
| `LEGACY_CONTEXT_MAX_MINOR` | `4` | the highest minor that keeps the legacy surface |

A module declaring minor ≤ 4 keeps `da_claim` under its existing contract (§9) and **MUST continue to
be admitted**. A module declaring minor 5 MUST export both of §17.2's exports; a missing or mis-typed
export is `BadModule` at the §1.3 front door.

### 17.2 `da_resource_plan` and `da_apply_execution_grant`

```
da_resource_plan(cfg_ptr: u32, cfg_len: u32,
                 capability_grants_ptr: u32, capability_grants_len: u32) -> u64   // (ptr << 32) | len
da_apply_execution_grant(grant_ptr: u32, grant_len: u32) -> u32                   // 0 == accepted
```

- **`da_resource_plan`** runs on the **capability-free assessment instance**, in the same position
  `da_claim` occupied: config and Capability Grants in, canonical CBOR out, instance discarded. It is
  a pure function of its arguments, MUST NOT call any capability import, and MUST be byte-identical
  across repeated invocations — two invocations that disagree are `ResourcePlanInconsistent`, which is
  deliberately **not** `ClaimInconsistent`: that names a mismatch between repeated physical-tier claim
  results, a different object with different semantics, and reusing it would equate the two.
- **Ownership (exact).** The returned span is **guest-owned**, obtained with exactly
  `da_alloc(len, 1)`. The host copies it out and frees it with the identical layout
  `da_free(ptr, len, 1)` **once**, even when the copied bytes are subsequently refused. Returning a
  slice into a differently aligned or excess-capacity allocation is non-conforming.
- **Bounds before read.** The host clamps the declared length against the plan's byte ceiling
  **before** allocating or copying. A ceiling checked after a copy has already paid for the copy.
- **Typed refusals.** `LogicalResourcePlanInvalid` for bytes that are not a well-formed schema-1
  plan — a zero or out-of-bounds span, malformed or non-canonical CBOR, unresolved names, or content
  naming a physical backend. `LogicalResourcePlanExceedsPolicy` for a well-formed plan breaching a
  declared bound: the byte ceiling, the node or dimension count, expression depth, or the derived
  derivation budget.
- **`da_apply_execution_grant`** is called **exactly once**, on the admitted run instance,
  **before `da_init`**. The span is **host-written and borrowed** on the existing config/grants
  convention: the guest decodes or copies it synchronously, retains no pointer, and never calls
  `da_free` on it; the host does not free it either, and it is reclaimed with the instance. A nonzero
  return is `ExecutionGrantRejected`, which records the module's `u32` verbatim and is deterministic
  and non-retryable for that `(module, plan, grant)` tuple — a retry requires changed admitted input,
  not a fresh instance.
- **The grant is not part of the Capability Grants.** Those are the bytes the plan was derived from;
  inserting the grant into them would make the grant an input to its own derivation.

### 17.3 The derivation budget is derived, not raised

Plan derivation is **capability-free, compute-free, allocation-free and execution-free**, so its fuel
is analytic rather than a constant somebody raised: a fixed base, plus a bounded cost per plan node
and per output byte, under an absolute ceiling. The ceiling is what can be armed before the plan
exists. If deriving a plan would require walking a materialized tensor graph, the plan representation
is wrong — symbolic formulas over bounded operation classes are the required form.

### 17.4 Minor-selected tag-0, and tag 18

At minor ≤ 4 the run header carries `claim`. **At minor 5 `claim` is FORBIDDEN** and the header
instead carries, inline: the canonical Logical Resource Plan, the composed role Physical Estimate,
the node/device aggregate estimate, and the Execution Grant — each with its blake3 digest. A record
carrying both a declared claim and a composed estimate is refused: a reader given both would have to
guess which figure the run was admitted on.

The grant is recorded **inline** rather than by sidecar reference on purpose: it is required before
initialization and before replay can execute guest code, so resolving it from a sidecar would create a
dependency at the very point that establishes execution identity. The encrypted-sidecar convention for
large readback values is unchanged.

**Tag 18** is the grant-application result: `{execution_grant_hash, status}`, written **exactly once**
after `da_apply_execution_grant` returns, whatever the status. Exactly one tag-18 record **or** one
terminal grant-application trap follows tag 0, and **replay reproduces the branch**. A replay verifies
every recorded digest against the bytes recorded with it before using any value, and checks that the
members agree about each other — the estimate names the plan it prices and the grant that configured
it, and the grant names the plan it configures.

### 17.5 The typed execution context (replaces §7.6's enumeration)

The trap context is a **closed domain of eleven values**, rendered **minor-selected**:

| Context | Canonical string |
|---|---|
| initialization | `da_init` |
| before the run loop | `da_run:before` |
| between slices | `da_run:between` |
| after the run loop | `da_run:after` |
| a slice | `slice:<canonical u64>` |
| the legacy claim export | `da_claim` |
| the manifest export | `da_manifest` |
| migration | `da_migrate` |
| the assessment instance | `assessment` |
| plan derivation | `da_resource_plan` |
| grant application | `da_apply_execution_grant` |

Rendering is selected from the **negotiated module ABI** when live, and from **tag 0's recorded ABI**
when replaying, so a replayed record renders as it was written. At minor ≤ 4 the historical `da_run`
string is preserved as **opaque legacy evidence** and is never re-rendered into one of the four
`da_run:*` refinements. There is **no twelfth value**: a pre-guest bring-up failure is a host-side
refusal with its own stage (architecture §6.5 [NC-5]), outside this domain entirely, and inventing a
context for it would state a fact about a phase the guest never entered.

### 17.6 `sys@2::log` during `da_init` and `da_migrate` (extends §6.5)

`log` is **exempt** from `PhaseViolation` during `da_init` and `da_migrate`. A module that cannot log
during initialization cannot explain why initialization failed, which is the moment the explanation
matters most; the exemption is narrow and applies to `log` alone.

The exemption is **bounded, and the bounds are ordered**:

| Bound | Value |
|---|---|
| calls per phase | `LOG_CALLS_PER_PHASE_MAX` |
| bytes per phase | `LOG_BYTES_PER_PHASE_MAX` |
| bytes per message | `LOG_MESSAGE_BYTES_MAX` |

The accepted prefix length is computed **from the arguments alone** — `log_accepted_prefix_len(raw_len,
remaining_phase_bytes)` — and the clamp is applied **before** any allocation or copy of the guest's
span. A reader that copied the whole span and then truncated has already paid for the untrusted
length. Panic and log text is **untrusted**: it is escaped at the sink, never parsed for
classification, and never used to decide a refusal.

### 17.7 The lane's sanity bounds are profile-keyed

At minor ≤ 4 the participation lane's claim bounds are checked against the module's declared
`claim.device_total()` / `host_total()`, unchanged. At minor 5 the lane's **profile-keyed** sanity
bounds are applied to the **composed role Physical Estimate**, after composition and grant binding
and **before** capability comparison and owner authorization, refusing
`PhysicalEstimateExceedsLane`. A lane that states no bounds for the backend class the profile prices
refuses `LaneProfileUnsupported` — silence is not permission. The order is load-bearing: an estimate
absurd for the lane is a lane violation, and reporting it as a machine that is too small sends an
operator to inspect hardware for a fault that lives in a plan or a profile.

### 17.8 Permitted revision ranges name their numbering

A profile's permitted revision range is evaluated against a record's revision fields, and those fields
carry **different numbering on different platforms**: `os.build` is the kernel release on Linux and a
wholly different OS build identifier on macOS. A range MUST name the platform whose numbering it
constrains, and a record from another platform's numbering does not satisfy it. See architecture §9.7
for the full caution; it is repeated here because this is where an author writes the range.
