# The VHC Fleet Ceremony — operator runbook

**Status:** the operator tooling described in §3 exists in this repository, reviewed and gated. This
runbook is the successor of the pre-refactor `docs/specs/swarm-p2-gate-runbook.md`; it documents how
the one hardware-in-the-loop validation of the product path is driven, end to end.

**Normative companions (in this repo / tree):** `docs/specs/vhc-architecture-spec.md` (architecture),
the VHC ABI/wire spec, and the streaming det-fold substrate spec. Where this runbook is terser than
those, they win.

**Path convention:** repo-relative paths resolve against this `daemon-node` checkout. The ceremony
is driven from a single operator box (the Linux build host); remote fleet boxes only run a `daemon`
node and receive `daemon-cli` commands over ssh.

---

## 1. What the ceremony proves — the gate

The ceremony PASSES iff all of the following hold over one full run on the exact frozen candidate:

| # | Criterion | Measured how |
|---|---|---|
| G-1 | ≥3 heterogeneous trainer peers over WAN — AMD RADV/Vulkan (Linux), Apple Metal (macOS), NVIDIA wgpu-DX12 (Windows): three GPU vendors, three OSes | fleet roster (§2) |
| G-2 | Zero det-digest mismatches: every peer that completes a round reports a byte-identical round digest, every round | the per-round digest transcript over the product API (§5.4) |
| G-3 | Churn survived: one hard-kill drill (worker SIGKILL mid-run → node respawn → checkpoint restore → rejoin → digest agreement resumes) and one graceful leave/rejoin drill | drill procedure (§5.5) |
| G-4 | Checkpoint plane live: remote checkpoints published at the genesis cadence; the churn rejoin restores from one | registry checkpoint pointers + restore logs |
| G-5 | Replay oracle green: the archived journals re-derive the run offline, byte-identically | the replay/verdict command (§6.2) |
| G-6 | Product path only: every peer is a full `daemon` node process, driven exclusively through the node's public API socket — no harness library calls, no test fixtures | drive model (§2.3) |

Any det-digest mismatch is a hard fail: stop, archive everything, replay offline to localize (§6.2).
It is never "re-run and hope". The convergence and round-overhead research criteria are explicitly
out of scope: this is a structural proof (digest agreement, churn, restore, replay), not a scale
study. Loss values are recorded as evidence but adjudicate nothing.

### 1.1 The freeze record is two layers, and every artifact cites both

*Owner-approved 2026-07-26 (the T7 amendment). This is what an operator has to do differently, and it
is not optional: an artifact citing one digest instead of two is incomplete and is not admissible
evidence.*

The candidate is frozen by one full `vhc-production-gate --all` battery on the merge to
`vhc-integration`. The freeze record has **two layers**, because the ratified resource model introduced
artifacts whose lifecycles are deliberately independent of the guest and of the code freeze — a Backend
Execution Profile, a Device Capability Report, a composed Physical Estimate, an Execution Grant. Putting
those in the candidate tuple would force a full re-freeze for a driver update; leaving them out
entirely would leave recertification triggers with nothing to compare against.

**Layer 1 — the candidate tuple (slow, code-frozen).** One ledger entry, one digest, pinning
together: the node commit; the cloud commit + deployed version id; the `guests.blake3` module hashes;
the tooling binary hashes; the **planner version identity**; the **governor implementation/version**;
the **full ABI `{major, minor}`** — not the major alone, since the certification minor carries the
pre-loop diagnostic semantics and both new exports together; and the **genesis schema major**. The four
added members are **contract identifiers, not release versions**: they move under their own contract
rules and never through a `VERSION` file. A change to any member re-freezes the tuple and re-runs the
full battery, exactly as before.

**Layer 2 — the composition evidence record (per backend, per fleet box).** One record per
`(participant, backend, admitted role instance)`, binding **twelve** members: role/incarnation
identity; participant/device identity; profile digest **+ profile authority**; capability-report
digest; logical resource plan hash; the Execution Grant's canonical-bytes digest; the composed role
Physical Estimate and the node/device aggregate estimates; planner identity; governor identity;
`reservation_identity`; a canonical **reservation digest**; and **scope-separated reservation
components** including the profiled hidden-overhead reserve as its own visible component, with directly
enforceable and profiled-and-measured amounts separated.

The last three are not optional: the earlier members bind the reservation's *inputs*, so an auditor
holding only those can recompute what should have been charged without learning what was. Identity
alone locates a reservation; it does not prove what was charged. And totals without scope separation
cannot distinguish a correctly-shared process-scoped term from a double-counted per-role one, which is
the arithmetic error the package exists to prevent.

A layer-2 record **never forces a battery by itself** — a profile correction, a capability-probe fix, a
grant reselection or a backend implementation revision produces a **new record** and leaves the guest
hash untouched. The two triggers that *do* reach layer 1 are a planner fix and a governor fix, which is
exactly why layer 1 now carries those two identities.

**The co-citation rule.** Every artifact produced after the freeze — certification statement, journal
evidence, replay verdict, preflight or conformance record — cites the **pair**
`(candidate_tuple_digest, composition_evidence_digest)`. **The candidate tuple never points forward**
at composition records: a frozen artifact cannot reference evidence created after it was frozen, so the
join is made by the citing artifact at the time it is written. An artifact spanning several participants
cites one candidate-tuple digest and the **set** of composition-evidence digests it covers, naming each.

**Cross-layer agreement is checked, not assumed.** A record's planner and governor identities MUST
equal the candidate tuple's, and a profile whose named compatible planner version excludes that planner
is not composable. A mismatch is a validation failure of the citing artifact, not a note.

**Append-only.** Records are never edited or deleted. Each carries `supersedes` (the digest it
replaces, or null), the trigger reason in plain language, and the consequence class that fired. A
**superseded, revoked or dangling** record fails a citing statement **closed** — which is why the
encoder/validator lands before any evidence is emitted under this regime.

---

## 2. The topology

```
                       Cloudflare (dev deployment)
                 ┌───────────────────────────────────────────────────┐
                 │ registry: POST /runs, envelope in R2, presign      │
                 │ seat CAS ─ roster CAS ─ checkpoint pointers        │
                 │ WS /runs/:id/ws  = byte-opaque dissemination relay │
                 └───────────────▲───────────────▲───────────────▲───┘
                        WS + HTTPS│              │               │
   ┌──────────────┐      ┌────────┴─────┐  ┌─────┴──────┐  ┌─────┴────────┐
   │ relay box    │      │ build host   │  │ Mac (Metal)│  │ Windows GPU  │
   │ iroh-relay   │◄─────┤ daemon node  │  │ daemon node│  │ daemon node  │
   │ only         │ iroh │ · trainer    │  │ · trainer  │  │ · trainer    │
   │ (no trainer) │ plane│ · COORDINATOR│  │            │  │  (wgpu-DX12) │
   └──────────────┘      │   seat holder│  └────────────┘  └──────────────┘
                         │ · operator   │        ▲                ▲
                         │   seat (ssh) ├────────┴────────────────┘
                         └──────────────┘   ssh → daemon-cli vhc <verb>
                                            against each box's local socket
```

- **Registry / relay / CAS / checkpoint pointers:** the dev cloud deployment.
- **Coordinator:** the build-host node claims the coordinator seat (a fenced lease in the registry
  CAS) and runs the genesis-pinned `coordinator_quorum.wasm` role. That box is also a trainer (two
  role instances, exactly the acceptance suite's proven shape).
- **Trainers:** the build host (wgpu/RADV Vulkan), the Mac (Metal), the Windows GPU box (wgpu-DX12).
- **Relay box:** iroh relay only.
- **Control planes:** dual — WS to the dev relay + iroh gossip via the relay box.
- **Payload plane:** R2 via the registry presign base.
- **Operator seat:** the human + this runbook, on the build host, driving every box over ssh with
  `daemon-cli`.

### 2.1 Why the coordinator is a node seat, not the cloud shell

The frozen genesis pins `authority = SingleKey(trusted_bases[0])` and
`Identities.coordinator = Some(trusted_bases[0])` — the authority identity must be known at
authoring. The cloud shell's identity is minted randomly at run init and only *returned* by the init
call, so a shell-coordinated run with a pre-authored pinned authority is circular. Peers verify every
frame's signer against the genesis authority, so any frames the cloud shell emits under its random
identity are dropped as unauthorized — inert noise, asserted in the single-peer smoke.

### 2.2 The fleet

| Box | Hardware | Role | Backend lane |
|---|---|---|---|
| build host | AMD, large UMA, RADV | trainer + coordinator seat + operator seat | wgpu/Vulkan |
| Mac | Apple, 32 GiB unified — the memory floor peer | trainer | Metal |
| Windows GPU box | NVIDIA, 32 GiB VRAM | trainer | wgpu-DX12 |
| relay box | small | iroh relay only | — |

Fit, from the frozen arithmetic (`crates/vhc/host/daemon-vhc-testkit/src/ceremony.rs`): device
working set ≈ 11.72 GiB fits every trainer; host retained det-state ≈ 14.65 GiB is disk-backed
everywhere; the largest tensor (192 MiB) clears the ~2047 MiB per-buffer clamp of the RADV and DX12
lanes. Budget ≈ 40 GiB free per trainer (corpus + retained-state spill + checkpoint scratch),
checked at preflight.

### 2.3 Why the drive is `ssh → daemon-cli` per box

The ssh transport carries no protocol — it just executes a local CLI on the box, which talks to the
local API socket (a Unix socket on Linux/macOS, a named pipe on Windows). Every verb maps 1:1 onto an
existing node API request, so there is no bespoke client to review, and commands are loggable,
replayable, and identical on every platform. Driving remote library calls would bypass node
identity/admission/lifecycle and fail G-6.

---

## 3. The operator tooling (it now exists)

The tooling below was implemented, reviewed, and gated as part of preparing this candidate. It is the
direct product-path replacement for the pre-refactor harness that owned envelope authoring, run
creation, join, and digest collection with library calls.

### 3.1 `daemon-cli vhc` — the drive surface

`bins/daemon-cli/src/cmd/vhc.rs`. Every verb marshals one existing node API request (no wire change)
and supports `--json` for stable machine-readable output; `identity` reads the local keystore with no
wire call (works with the node stopped).

```
daemon-cli vhc runs [--json]                              # list discovered/joined runs + eligibility
daemon-cli vhc detail <run-id> [--watch <secs>] [--json]  # snapshot; --watch prints phase, round,
                                                          #   last-round digest, and peers each poll
daemon-cli vhc join <run-id> [--policy always|idle|manual] [--json]   # a fresh idempotency op_id is minted
daemon-cli vhc leave <run-id> [--immediate] [--json]
daemon-cli vhc pause <run-id> [--json]
daemon-cli vhc resume <run-id> [--json]
daemon-cli vhc hardware [--json]                          # this node's training-capability probe
daemon-cli vhc identity [--state-dir <dir>] [--json]      # the local base-identity PeerId (hex)
```

`identity` resolves the keystore at `<state-dir>/vhc/identity` (or `$DAEMON_VHC_IDENTITY_DIR`, else
`$DAEMON_DATA_DIR/vhc/identity`). The base-identity secret never leaves its box; only the public
PeerId is collected for the genesis trust set / roster.

There is deliberately NO `switch` verb: the live module-switch drill was dropped by adjudication
(post-switch cross-peer round progression is proven nowhere), and the switch request already has its
own acceptance gate.

### 3.2 `cargo run -p xtask -- author-ceremony-genesis` — genesis authoring

A thin wrapper around the frozen, reviewed library (`daemon_vhc_testkit::ceremony::ceremony_genesis`
— never reimplemented). It also authors single-peer smoke geneses (`--min-peers 1 --max-peers 1
--stop-rounds <small>`).

```
cargo run -p xtask -- author-ceremony-genesis \
  --run-label <s> --author-key <file|hex> \
  --coordinator-module <blake3-hex> --trainer-module <blake3-hex> \
  --coordinator-wasm <path/to/coordinator_quorum.wasm> \      # the FILE, checked against its digest
  --trainer-wasm <path/to/tiny_llama.wasm> \                  # the FILE, checked against its digest
  --corpus-manifest <path/to/corpus-manifest.cbor> \
  --trusted-base <hex> --trusted-base <hex> --trusted-base <hex>   # ORDERED; first = authority
  --roster <hex> --roster <hex> --roster <hex> \
  --upgrade-authority <hex> \
  --min-peers 3 --max-peers 3 --ckpt-cadence 4 --payload-retention 64 \
  --warmup-s <n> --round-max-s <n> --witness-s <n> --cooldown-s <n> --stop-rounds <n> \
  --out <dir>
```

`--ckpt-cadence` is bounded by the retained record horizon (16 rounds,
`RETAINED_RECORD_HORIZON_ROUNDS`), not only by retention: a rejoiner's fence trails the live head
by up to one full cadence slot, and the coordinator's in-memory ring replay bridges at most the
horizon. The authoring refuses a wider cadence typed since defect 7c — but since Gate B' this
inequality is **sizing for the ordinary rejoin, not the recoverability guarantee**: a fence that
trails past the ring is bridged by **staged archive catch-up** (the node rides the verified
archive lineage in the join credentials; the worker extracts the coordinator's historical round
records from attested segments and folds them before live attach; round-aware seal pacing in the
session keeps the archive tip within ring reach of the live head). `CheckpointStale` now refuses
only the gap BOTH planes genuinely cannot bridge — an archive tip itself beyond ring reach of the
head (a publication outage outlasting the ring; Gate C keeps publication retrying budget-free).
The c15f drill shape (cadence 8, fence 16 rounds behind) recovers through catch-up today.

**Payload retention honesty:** `--payload-retention` is an authored claim, not an enforced R2
lifecycle — no production path deletes R2 payloads by age today (only the fs payload plane
prunes). Ceremony verification must therefore *prove* payload availability across the run rather
than assume the window. When payload GC ever lands, it must pin the **recovery closure of the
latest usable checkpoint** (every payload a catch-up from that fence could fetch), never delete
by nominal age alone.

**The two module FILE flags are required, and they are not a convenience.** `--coordinator-wasm` and
`--trainer-wasm` are checked against `--coordinator-module` / `--trainer-module`, and the file is needed
because a role's **execution requirements are derived by running the module's own assessment export**
(architecture §5.4 [DI-1], §9.6 [RC-1]). A digest cannot be asked what it needs. Authoring has no
constructor for stating a requirement by hand, so an authoring run without the files cannot produce a
role entry at all — and that is the intended shape: the module is the only authority on its own
resource plan.

The real run timers were added to the library's genesis spec with defaults equal to the prior
synthetic-clock values, so an untuned authoring is byte-identical to before; only tuned timers move
the run id. Genesis-rule violations (profile-chunk divisibility, state-chunk validity, checkpoint
cadence vs retention) refuse at authoring.

Outputs into `--out`:

- `envelope.cbor` — the canonical `SignedEnvelope` **wire form** (`{ bytes, signature, signer }`)
  wrapping the frozen genesis: this is the object the registry stores and the node's assess path
  decodes (`from_canonical_slice::<SignedEnvelope>` → `FrozenGenesis::open`). Seed it verbatim —
  do NOT unwrap it to the inner genesis (the pre-fix tool emitted the raw inner bytes, which made
  `vhc join` refuse `UnsignedEnvelopeRetired … missing field bytes`). **Inner vs wire:** the run
  id is `blake3(inner frozen genesis)`, NOT the blake3 of this wire object (the latter is the
  registry descriptor's `envelope_hash`);
- `envelope.b64` — base64 of those exact `SignedEnvelope` wire bytes, for the cloud seeder's
  `VHC_ENVELOPE_B64`;
- `run-id.txt` — the genesis hash hex (blake3 of the inner frozen genesis — the cryptographic run id);
- `authoring-report.txt` — every frozen pin (param count, expected root, chunk sizes, cadence check,
  timer values, module + corpus hashes, trust set, roster, upgrade authority) restated for human
  ratification.

### 3.3 Corpus + module publication (pre-existing)

- `cargo run -p xtask -- tokenize-corpus …` — tokenize + chunk-address the corpus, emitting shards +
  `corpus-manifest.cbor`.
- `cargo run -p xtask -- publish-corpus --manifest … --run <run-id> …` — upload shards + tokenizer +
  manifest to `runs/<run>/corpus/…` by content hash.
- `cargo run -p xtask -- publish-module --module … --run <run-id> …` — upload a module to
  `runs/<run>/modules/<blake3>.wasm`. Module fetch is content-addressed (`modules/<blake3>.wasm`) and
  blake3-verified; the envelope's artifact `url` note is the honest content-addressed
  `r2://modules/<blake3>.wasm` form (the run id is not knowable while authoring the envelope that
  defines it).

### 3.4 `cargo run -p xtask -- vhc-replay` — the replay/archive verdict

Runs BOTH oracle modes over an on-disk archive and emits a per-round, per-peer verdict.

```
cargo run -p xtask -- vhc-replay --archive <dir> --run <run-id> [--json]
```

- **Consensus re-derivation (sandboxed, cross-chain certified).** The pinned coordinator module is
  driven inside the real host sandbox — consensus never runs natively — over the archived driving
  inputs recovered from the sealed record segments; every archived round record must re-derive
  byte-identically, and every committed digest must recompute from the content-addressed payloads
  alone. A lineage spanning MULTIPLE chains (restart succession) routes through the certification
  kernel: every signed head is bound to its segment bytes (sealed + every identity field + the
  seal's pre-seal record count), every reason-2 seam is bound (the successor's anchoring tag-10
  manifest must equal the predecessor replay's exported capture; every recorded kind-3 restore
  read-back must equal the section staged at that identity, by value or sidecar hash), and the
  lineage-global semantic fold refuses equivocation (two different `RoundRecord`s for one round),
  continuity breaks (the deduplicated committed rounds must be dense from 0 — identical
  replay-forward re-publishes deduplicate and count once), and conflicting per-peer digests.
- **Per-peer digest agreement.** Each peer's recorded per-round det-state digest is compared across
  peers; a round where any peer's digest differs from the round quorum is a disagreement, and the
  earliest such round is the first divergence.

The verdict is GREEN iff the consensus oracle agrees AND (when peer transcripts are present) every
per-round digest agrees across every reporting peer; otherwise RED, carrying the failing stage and
the first divergence round. GREEN additionally reports the **closure class**:

- **`terminal(<outcome>)`** — the final span's replay rode the recorded stop into the guest's own
  `da_run` return and reproduced the recorded tag-9 kind-0 outcome: the archive is a COMPLETE
  record of a finished run.
- **`prefix`** — a verified sealed prefix (an archive is a sealed prefix by construction; a
  still-running, killed, or terminal-head-less lineage certifies here). Every recorded decision is
  still verified; only COMPLETENESS is not claimed.

Consumers assert on the class: the C2 evidence gate (§6.2) requires `terminal` on the completed
run's archive; a `prefix` there means the terminal segment's head is missing and must be
explained before any completeness claim.

#### Archive directory layout contract

```
<archive>/
  envelope.cbor                     the frozen genesis envelope bytes (authority + the pinned
                                    coordinator module hash; its blake3 IS the run id)
  coordinator.wasm                  the genesis-pinned coordinator module (blake3 == the envelope's
                                    coordinator.wasm artifact hash)
  heads.cbor                        CBOR: [ <archive-head-record> ] — the run's published ABI §8.8
                                    head records (every role's chains; the reader authorizes each
                                    against the envelope's genesis-trusted bases and selects the
                                    coordinator lineage itself)
  segments/<segment_hash_hex>.seg   the sealed record-archive segment bytes (content-addressed; the
                                    file stem is the segment's blake3 content address)
  payloads/<blake3_hex>.bin         the committed update-container payload objects, by content hash
  peers/<peerid_hex>.digests.cbor   CBOR: [ [round, <16-byte digest>] ] — that peer's per-round
                                    post-ingest det-state digests (extracted from the coordinator
                                    chain's recorded signed `Digest` inputs)
```

`envelope.cbor`, `coordinator.wasm`, `heads.cbor`, and `segments/` are required (the consensus
oracle); `payloads/` is required for the digest-from-payloads re-verification; `peers/` is optional
(absent → the per-peer agreement section is empty, the consensus oracle still runs).

#### Where the archive comes from (the product path)

Segments and heads are published **live, per seal**, by every role session with a durable journal
home (ABI §8.8): segment bytes ride the run's content plane (presigned R2 / the shared filesystem
store) at their BLAKE3 address; each seal's attested head record lands on the registry's archive
slot (`PUT /runs/:id/archive/head`, snapshot via `GET /runs/:id/archive/heads`) — or, on a
registry-less run, under `<run state dir>/archive/heads/`. The registry applies the structural
dense-chain fold only (idempotent republish; a non-extending head is refused typed — fork
evidence); the assembler verifies every head's signature + certificate chain against the
genesis-trusted bases while pulling the layout above together. The rotation policy's five-minute
age bound is the recovery-point cadence: a live run's remote archive trails its journal by at
most one open segment span. Segments seal only on roll/terminal — an aborted process's unsealed
tail is recovered, sealed and republished by the next incarnation's startup reconciliation.

#### `cargo run -p xtask -- vhc-archive-pull` — assemble the layout from a live run

```
cargo run -p xtask -- vhc-archive-pull --run <run-id> --base <gateway base> \
    [--bearer <token> | --org <org> --actor <actor>] --out <dir>
```

Pulls the layout above from the registry + content plane with a third party's trust posture: the
envelope is blake3-verified against the run descriptor, every published head record is authorized
(`ArchiveHeadRecord::authorize` — per-run signature + certificate chain to a genesis-trusted
base), every chain is re-folded structurally, and every content object (segments, committed
payloads, the pinned `coordinator.wasm`) is fetched by content address and re-hashed on arrival.
Committed payload hashes are enumerated from the coordinator lineage's published `RoundRecord`s;
the per-peer digest transcripts are extracted from its recorded signed `Digest` inputs. The
verifying core is `daemon_vhc_observe::assemble_archive` — the same function the testkit gate
(`daemon-vhc-testkit/tests/archive_assembly.rs`) drives end-to-end: real sandboxed coordinator
run → per-seal publisher → untrusted stores → assembly → GREEN `vhc-replay` (that test keeps its
assembled layout under `DVHC_KEEP_ARCHIVE=<dir>` for a manual CLI smoke).

**The assembly is resumable (REL-2a, reliability spec §3.1).** An interrupted or failed pull
is re-run with the same `--out`: every content object already verified on disk (its bytes
re-hash to the requested address) is reused instead of re-downloaded, and the report separates
`fetched` from `reused (resumed)` counts per class — a resumed assembly is visible, never
silently claimed as fresh. A torn or tampered local file fails the same blake3 gate a fresh
fetch gets and is re-fetched and atomically repaired. Transient egress faults
(connect/timeout/reset/5xx) are absorbed per object by the store's bounded retry rather than
aborting the pass; a genuine 404 still refuses immediately.

A coordinator lineage that spans MULTIPLE chains (restart succession) assembles fine — every
chain's heads and segments land in the layout — and `vhc-replay` certifies it through the
session certification kernel (`certify_lineage`: the same executor the join-transaction rebuild
runs, under the `Certify` end policy), so a drill run carries the replay claim itself. The
single chain is the degenerate lineage through the same path; harness-form archives are
unchanged. Pinned by the `reconstruct_product` certification matrix and the fold/binding/seam
unit gates.

A succession may also cross **base identities** (the seat moved boxes): the publisher's
succession-link resolution considers a predecessor chain published under a DIFFERENT
genesis-trusted base — a foreign-base candidate must `authorize` against the trusted set before
it may shape the founding head's link (an unverifiable store row never does) — so the reader's
lineage fold still sees exactly one founding chain per seat. Own-base linking needs no trusted
set; the cross-attestor case is pinned by the `reconstruct_product` matrix
(`archive_only_recovery_reconstructs_across_a_second_trusted_base_identity`).

**The crash-tail closure (defect 16, c15k).** A hard-killed coordinator leaves a suffix of
records past its last archived head — sealed segments the outage kept from publishing, plus the
unsealed cut segment. Reconstruction consumes that suffix (the successor's boot capture folds
it), so it MUST reach the archive or every later archive-only fold replays a state *behind* the
successor's recorded restore read-back and refuses at the content-address gate. Two mechanisms
close the gap on the production path:

- **Seal before consume** — `recover_records` runs `seal_abandoned_tail` over each lineage
  chain's local journal home before walking its tail: the crash-cut segment is truncated to its
  durable length and sealed in place, becoming publishable archive material. A tail that cannot
  be sealed is NOT consumed (its records could never reach the archive).
- **Predecessor backlog adoption** — the successor session's publisher receives the recovery
  lineage's predecessor chain instances (`ArchiveSpec.predecessors`, derived from the
  `reconstruct` credential) and, before its own founding head commits, uploads every
  sealed-but-unpublished predecessor segment and attests its head under the successor's OWN
  certified span (cross-span attestation: `body.chain_instance` = the predecessor's,
  `body.instance` = the successor's — the reader's span-monotonicity rule `instance ≥
  chain_instance` admits it). The succession link then names the COMPLETE predecessor terminal.

Both are pinned end-to-end by
`reconstruct_product::a_consumed_crash_tail_reaches_the_archive_and_the_lineage_reconstructs_archive_only`.

### 3.5 The per-round digest on the product API

The per-round det-state digest each peer produces (§5.6 of the architecture) is surfaced on the node
API: `VhcEvent::RoundOutcome` carries a `digest`, and `VhcRunDetail` carries `last_round_digest` so a
polling client reads the newest round's digest without an event subscription. Both are additive; the
contract wire revision was bumped one rung and the conformance suite validates the new shapes. The
node no longer drops the digest at the conversion boundary, and the snapshot projects the newest
round's digest for the `detail --watch` transcript.

**Digest origin (opacity-safe live producer).** On the live run path the digest reaches the API
without any host-side frame decoding. The trainer guest already computes its per-round det digest and
voices it as its own `[4, round, digest]` det-lane publish (the journal transcript); it ALSO reports
that same digest — plus the barrier's `committed`/`ingested`/`stalled` bookkeeping — through the host
metric ABI (`sys@2::emit_metric`) under the reserved `round_metrics` name contract (a group of
`vhc.round.<round>.<field>` metrics; the 16-byte digest rides as four little-endian `u32` words, each
lossless in an `f64`). The role session recognizes those reserved metric NAMES only — it never decodes
the module's `[tag, round, bytes]` frame vocabulary — and folds each round's completed metric group
into a `RoundOutcome` session event, which the node projects as above. The guest reports honest,
guest-known values: at the barrier every record-listed committed payload has been fetched and folded,
so `committed == ingested` (the record-listed set size), and `stalled` marks a round that straggled and
caught up. The reserved contract is defined once in `daemon-vhc-abi` (`round_metrics`), pinned by both
the guest SDK emitter (`daemon_vhc_sdk::report_round_outcome`) and the host session recognizer.

---

## 4. Preflight (per item: commands → evidence → remedy)

Order matters and is deliberate:
`cloud/relay → binaries/guests → identities → corpus → device smokes (timer calibration) → genesis
authoring → seeding/publication → seat proof`. The corpus precedes the smokes (a smoke without the
real tokenized corpus calibrates fetch-free round walls), and the smokes precede the genesis (the
measured round wall on the slowest box sets the immutable run timers).

### 4.1 Registry deployment

Re-auth wrangler with the human if expired; deploy dev at the cloud head; run the live smoke
(create/state/wss-broadcast/presign round-trip). Record the deployed version id + the cloud git rev.
During the first single-peer smoke, grep the trainer log for rejected unauthorized coordinator frames
and assert round progress is unaffected (the cloud shell is inert). Keep the deployment in sync with
the merged cloud branch at the candidate freeze.

### 4.2 The iroh relay

Probe the relay's `generate_204`. If dead, launch it per `crates/vhc/host/daemon-vhc-net/dev/`
(pin the iroh-relay version it wraps), verify the 204, record the relay URL + version. Relay
placement is not part of the gate — only the dual plane's existence is; any box (including the build
host) can host it.

### 4.3 Binaries + guests, per platform

Resource discipline throughout: one local build at a time, jobs capped at ≤ nproc/2.

1. **Guests (build host, once):** `cargo run -p xtask -- build-guests`; assert the produced
   `guests.blake3` equals the committed pin (reproducibility gate — on mismatch, stop). Record the
   `tiny_llama.wasm` + `coordinator_quorum.wasm` blake3 hashes.
2. **Build host:** devshell debug-with-opt build of `daemon`, `daemon-cli`, `daemon-vhc-worker` with
   the wgpu feature (the acceptance suite's byte-identical rebuild discipline is the template).
3. **Windows (sealed cross-build, never on-box):** cross-build the `daemon`, `daemon-cli`, and worker
   binaries + copy `tiny_llama.wasm`. Probe `daemon-cli.exe vhc hardware` against a started node.
4. **Mac (on-box build):** sync the candidate tree (excluding `target`, `.git`, `result*`), build
   inside the devshell with the login-PATH export; pre-realize the dev env so node/worker spawns are
   instant.
5. **Provenance:** each deployed binary's `blake3` + the candidate commit recorded per box. Disk
   check: ≥ 40 GiB free per trainer.
6. **Featured build check (CHECKED, per worker binary):** the worker MUST be built with its
   platform lane's features (`--features vhc-net,wgpu-spirv` on the build host / Windows DX12
   lane / Mac Metal lane) — a featureless worker builds fine and then refuses
   `BackendUnavailable` at join. Evidence: the build command line recorded beside the binary
   hash, AND backend availability verified ON-BOX (`daemon-cli vhc hardware` names the expected
   backend) — the stale-image lesson: never trust the build host's view of a shipped binary.
7. **Profile re-provision check (CHECKED, after EVERY worker rebuild):** a rebuild changes the
   sealed backend revision, which invalidates every dev profile — run `vhc-provision-dev-profile`
   and re-run the fit probe on each box whose worker changed, BEFORE any join. Evidence: the
   fresh probe verdict (GREEN, key digest recorded). Skipping this refuses
   `EstimateNotComposable` at join (typed, correct — but a preflight failure, not a run finding).

### 4.4 Keystores + the trust set

On each trainer: write the `[vhc]`-enabled node config (§5.1), start the node once, stop it; run
`daemon-cli vhc identity` → the box's base-identity PeerId. Collect the three PeerIds: the ordered
trust set (build host first — it is the authority), the roster (all three), and the upgrade authority
(owner decision; default recommendation: the build host only, matching the frozen test shape). Base
secrets never leave their boxes. The genesis author key is a fresh operator key minted on the build
host and archived in the ledger (public half only).

### 4.5 Corpus (before the smokes)

Tokenize the ratified corpus (TinyStories under the TinyLlama SentencePiece tokenizer) with
`tokenize-corpus`, at `--seq-len 512 --token-width u16` (the frozen `CEREMONY_SEQ_LEN`). Resolve +
record the exact dataset +
tokenizer revision SHAs at this step (placeholders are forbidden past here). Record the
`corpus-manifest.cbor` blake3 — it is a genesis input, and the smokes consume the same corpus so
calibration reflects real data-plane behavior.

### 4.6 Single-peer smokes + timer calibration (after corpus, before genesis)

Per trainer, a throwaway single-peer run authored with the same authoring tool (its own label,
`--min-peers 1 --max-peers 1`, small `--stop-rounds`, the real corpus manifest), seeded to the dev
registry, its corpus + modules published under the throwaway run's prefix, through the full product
path:

```
daemon-cli vhc join <smoke-run>
daemon-cli vhc detail <smoke-run> --watch 10
```

Pass: ≥ 2 completed rounds with digests visible in the `--watch` output (the API digest surface
exercised end to end); the seed-init expansion cross-checks the pinned expected root (a typed init
failure is a hard stop); real corpus chunks fetched + verified over R2; device fit confirmed; the
state store spills to disk (the Mac especially); a checkpoint seal + publish + pointer visible in the
registry. **Measure the per-round wall clock on the slowest box** — the ceremony timers are set from
it.

### 4.7 Author, verify, seed, publish, fetch-verify (the committing step)

Strict order, because the run id (= genesis hash) commits everything and prefixes the publication
keys:

1. **Author** with the module hashes, the corpus manifest hash, the trust set / roster / upgrade
   authority, `--min-peers 3 --max-peers 4`, `--ckpt-cadence 4 --payload-retention 64`,
   `--stop-rounds 48`, and timers from the smoke calibration (round-max ≈ 3× the slowest measured
   wall; warmup generous). Review `authoring-report.txt`; the human ratifies it before seeding.
   **Churn headroom is MANDATORY (`max_peers = min_peers + 1`), not tuning.** With a FULL fixed
   roster (min = max), a crashed trainer's roster entry stays healthy-but-absent for `k_absences`
   rounds while its rejoined incarnation's every Join rejects `RosterFull`; the eventual drop then
   breaches the membership floor into `WaitingForMembers` (no timeout) with nothing pending, and
   the frozen guest's announce loop has already stopped on observing round traffic — a deadly
   embrace that idles the run indefinitely (found live in the three-seat smoke, 2026-08-09; the
   full mechanism is in that run's VERDICT). G-3's kill+rejoin drill re-creates exactly this
   shape; the churn slot admits the new incarnation while the zombie decays. The authoring
   cadence check already budgets one churn slot against retention.
   **Cadence wiring check:** `--ckpt-cadence` must land in the trainer role's `live` config as
   `remote_ckpt_every` (ABI §12.14 [SF-6] wiring note) — a validated-but-unwired cadence silently
   runs the guest's serde default of 0 (upload at every boundary), so the G-4 gate would never
   exercise the authored policy. The authoring path wires it since the Phase-1 cadence fix
   (regression: `ceremony_authored_round.rs::the_authored_checkpoint_cadence_reaches_the_trainer_config`);
   confirm the value in `authoring-report.txt` alongside the other pins. The harness/assessment
   config form stays `live`-free, so the cadence never moves the fit-probe key (no re-probe on a
   cadence-only change — but the envelope bytes and hence the RUN ID always change).
   **Seat binding check:** the authoring emits one trainer role **per roster seat**
   (`trainer-0`, `trainer-1`, …), each `RoleEntry` binding its seat's base identity
   (`identity`) and freezing that seat's plan identity (`peer`) in the opaque config. A joining
   node selects the seat bound to its own base identity; an undirected join against a seated
   genesis is a typed refusal (worker-side backstop). This is the defect-6 fix from the c15
   drills: a single shared trainer role decoded `peer = roster[0]` on **every** box — all boxes
   trained the same window slice (the rest of the authored global batch untouched) and the
   checkpoint slots elected to any other roster identity were published by nobody (the G-4
   round-16 hole). All seats share the ONE derived trainer execution requirement, so the
   fit-probe key does not move with the seat.
2. **Verify locally:** the tool re-opens the frozen envelope (re-derives the hash + verifies the
   signature) and reproduces the expected root before writing the artifacts.
3. **Seed:** `VHC_BASE=<dev base> VHC_ENVELOPE_B64=$(cat <out>/envelope.b64) node
   apps/vhc/scripts/seed_run.mjs <run-id>` → 201; GET `/runs/:id` + `/state` → 200.
4. **Publish** under the now-known run prefix: `publish-module` for both modules; `publish-corpus`.
5. **Fetch-verify from a different box** (the Mac): `daemon-cli vhc detail <run-id>` resolves the run;
   the node fetches + blake3-verifies the envelope — proving discovery end-to-end before anything
   joins.

### 4.8 Seat-claim proof

The build host joins first; assert the registry seat slot holds its lease and the heartbeat updates.
This is the last preflight gate; after it the run is live and waiting at the membership floor.

---

## 5. The run

### 5.1 Per-box node config (template)

```toml
# daemon.toml — ceremony trainer
[vhc]
enabled     = true
worker_path = "<abs path to daemon-vhc-worker[.exe]>"

[vhc.registry]
base = "https://<dev registry base>/api/v1/vhc"
auth = { internal = { org_id = "org_ceremony", actor = "key:<box-label>" } }

[vhc.iroh]
enabled       = true
relays        = "http://<relay box>:3340"
bind_port     = 41414
advertise_ips = ["<box WAN/LAN ip>"]

[vhc.default_policy]
mode = "always"

[vhc.owner_budget]
disk_mb = 245760   # CEREMONY-SPECIFIC: 3x the 60 GiB seat reservation + slack (see below)
```

**The 240 GiB owner budget is ceremony-specific configuration, not a production sizing rule.**
REPLACE-mode re-admission (tuple drift after any rebuild) reserves the incoming seat's FULL
fresh claim before the superseded seat releases, so a box holding two standing 60 GiB seats
needs one extra transient reservation's headroom or arbitration refuses and burns the retry
budget into `FailedTerminal` (observed live, c15m). The long-term fix is transactional
replacement accounting — charge only the DELTA while retaining the incumbent's fencing, never
release-before-reserve (releasing first would un-fence the incumbent while the replacement can
still fail) — recorded as future work; until it lands, ceremony boxes carry the 3× budget.

The coordinator allowlist on every box pins the dev registry base (the node refuses any other
coordinator endpoint). The build host's config additionally volunteers for the coordinator role (the
genesis names its base identity as the coordinator). Windows uses the same TOML; the named-pipe
socket is implied (`DAEMON_SOCKET_PATH` only if non-default).

`bind_port` is **one port per BOX**, and that holds on the box that also trains
(`coordinator_trains = true`). The iroh endpoint is a node-level identity — one keystore transport
key, one published roster record per run (architecture [CI-10]) — so co-located role-instances share
the box's single endpoint rather than binding a port each: the instance that joins first (the seat
instance on the build host) owns it, and its sibling attaches over the WS control plane every member
of the run is already on. The node says so explicitly (`co-located role-instance shares this node's
single iroh endpoint … attaching WS-only`); it is the intended topology, not a degraded run. Giving
a co-located instance its own port would publish reachability the box cannot honor (two live
endpoints, one endpoint id).

### 5.2 Evidence capture starts before bring-up

Start every box on **fresh vhc run state**. A persisted join intent from a superseded run rehydrates
at boot and is re-admitted against the owner ledgers before the node knows whether that run still
exists (the reservation is what converges the ledger to the genuinely-running set), so a stale
trainer intent holds a full accelerator-duty until its re-join resolves — bounded now (a refused
attach reports a typed terminal, releasing the duty and scheduling the retry; a failed re-authorship
no longer aborts the other intents' re-convergence), but still a noisy start. Either
`daemon-cli vhc leave <old-run>` first, or wipe the run state (`vhc.db*`, `vhc/runs/`,
`vhc/identity/runs/`) while PRESERVING `vhc/identity/base.key` — the genesis trust set names that
base identity.

On each box: node log to a file (`RUST_LOG=info,daemon_vhc=debug`); record the journal paths under the
node state dir (the primary replay artifact). On the build host, start the digest-transcript
collector — a loop over all three boxes' `daemon-cli vhc detail <run-id> --json` every 30 s appending
to `transcript-<box>.jsonl` (the digest rides the snapshot's `last_round_digest`).

### 5.3 Bring-up order (each step gated on the previous)

1. Relay probed; registry live; run seeded + artifacts published.
2. **Build host:** start node → `daemon-cli vhc join <run-id>` → seat claim asserted; the node admits
   both role instances (coordinator + trainer); watch `detail` until `WaitingForMembers` with 1 peer.
3. **Mac:** start node → join → roster record published; watch until 2 peers.
4. **Windows:** start node → join → floor reached (3) → warmup → round 0.
5. First checkpoint: assert a registry pointer at round 8 (the cadence).

From here the run drives itself; the operator's job is the transcript, the drills, and the abort
watch.

### 5.4 Digest agreement (G-2, continuous)

The transcript collector diffs per-round digests across the three boxes every poll. Any mismatch:
abort protocol — `daemon-cli vhc pause` on all boxes (best-effort), archive all journals + node
stores + the registry state + R2 record sets immediately, then §6.2 offline replay to localize. No
restart of the run.

**Plane liveness (deaf-path diagnosis).** Each role session reports a `plane_health` line
(a warning-class event in `daemon-cli vhc detail`'s recent events) on a slow cadence and at
session end: the transport counters (`ws[binary= hb= delivered= dup= deaf_reconnects=
last_binary_ms=]`) followed by the attach-boundary counters (`session[forwarded= delivered=
dup= held= refused= records= last_delivered_ms=]`), in delivery-chain order (ABI §12.8
[LT-5]). Reading it: a frozen `ws[binary]` while the socket stays connected is registry-side
deafness — the client detects it via the server heartbeat's expected-progress deadline and
force-reconnects (counted in `deaf_reconnects`); `ws` advancing while `session[delivered]` is
frozen localizes the loss between the dual plane and the attach (look at `refused`/`held`).
Repeated `deaf_reconnects` without recovery is a registry finding — archive and adjudicate,
never wait it out.

**Stall announcement (reliability spec §6, REL-5).** A joined, alive run whose committed
progress has aged past its per-run threshold (adaptive: 2× the largest observed inter-commit
gap, 10-minute floor) voices a `run_stalled` warning in `detail`'s recent events — once per
episode, closed by `run_progress_resumed` on the next committed round. The detail carries the
last committed round and BOTH ages (committed vs. local activity): committed old + local
fresh reads "the box is working but nothing is committing" (checkpoint publication, or the
quorum is parked — check the other boxes); both old reads "this box itself is idle". A
`run_stalled` that stands while other boxes advance is the §5.5-shape incident — do not wait
it out. Session `phase` values are now honest lifecycle states (reliability spec §6.1):
`restoring` and `catching_up` are healthy transits (watch them progress, minutes at ceremony
scale), `running` means genuinely attached and live, `draining` is a graceful leave in
flight — a box showing `restoring`/`catching_up` with a stale run head is catching up, not
wedged.

### 5.5 Churn drills (G-3/G-4)

- **Hard-kill drill at ~round 12** (first checkpoint exists at 8): on the Mac, `kill -9` the
  `daemon-vhc-worker` process (NOT the node — the node is the supervision layer under test). Expected:
  the node observes the worker death, respawns, re-admits under a new incarnation, restores from the
  live checkpoint, rejoins; digests agree from the restore round onward.
- **Graceful leave/rejoin drill at ~round 24:** on Windows, `daemon-cli vhc leave <run-id>`
  (graceful). With `min_peers = 3` the run parks below the floor — confirm the parked state via
  `detail` for a bounded interval (2× the calibrated round wall), then `daemon-cli vhc join <run-id>`
  → restore from checkpoint → resume. Zero digest disagreement across the seam. The rejoin is
  admitted through the genesis churn slot (`max_peers = min_peers + 1`, §4.7) — on a FULL fixed
  roster it would reject `RosterFull` until the zombie entry decays and the floor breach lands
  the coordinator in a timeoutless `WaitingForMembers` (the three-seat-smoke deadlock).
- **Coordinator crash drill** (ABI §8.8 [AR-8]): hard-kill the COORDINATOR's worker after at
  least one sealed segment has published (`GET …/archive/heads` non-empty). Expected: the
  node's rejoin resolves + verifies the head lineage, the fresh worker reconstructs the
  consensus state through the sandbox (log line "coordinator reconstruction: replaying the
  recovered lineage through the sandbox" with chain/frame counts), reports ready only after
  the replay drains, and the run resumes WITHOUT re-opening any round the durable record
  already committed — digest agreement from the next round onward. A join refused with a
  head-verification error is the fork/corruption fail-closed path: adjudicate the archive
  (§5.7), never force a fresh seat.
- **Seat-role leave/rejoin caution (three-seat smoke, 2026-08-09):** a `vhc leave --immediate`
  + `vhc join` cycle on the SEAT box inside the coordinator seat-lease TTL comes up
  TRAINER-ONLY, silently — `claim_now` stands down to the live-looking dead lease, and the
  resident keeper never retries a run whose admitted role is trainer. After ANY seat-box
  rejoin, VERIFY coordinator duty resumed (the "resolved coordinator reconstruction directive"
  log line, or a fresh `coordinator-<incarnation>` run directory) before trusting progress;
  if it came up trainer-only, wait out the lease TTL and cycle leave/join once more.

**Bring-up serialization (defect 17, c15k).** One bring-up transaction per run per node: an
explicit `daemon-cli vhc join` issued while the node's auto-resume reconvergence is in flight
refuses typed ("a bring-up … is already in flight — retry after it settles"), and a
reconvergence firing under a standing explicit join yields to it. Pre-fix both ran
concurrently and minted competing incarnations whose supersession left the survivor a zombie
— certified outbound, refusing every inbound record, no liveness signal. Defense in depth on
the session side: after every ingested distribution record the attach re-judges its OWN key
(`CertCheck::own_death`); evidence of self-supersession (a higher incarnation on the own
certifying base's ladder, an explicit revocation, or a fencing seat grant) ends the session
typed-retryable (`superseded` warning) so reconvergence mints a fresh incarnation above the
floor — supersession is a terminal, never a mute.

**Catch-up staging is gap-driven, not horizon-driven (defect 18, c15m).** The nominal
16-round retained ring is NOT a replay guarantee — a reconstructed coordinator's ring starts
at its boot round. The node stages archive catch-up for ANY restore-fence gap the verified
lineage usefully reaches (tip covers the fence, within a ring of the head); overlap with the
live ring replay is absorbed by the dedup window. Pre-fix, a within-horizon gap after
reconstruction staged nothing and the respawned trainer looped `OUTCOME_STALE_RESTORE`
through the paced-respawn lane indefinitely. A within-ring gap with no published archive
proceeds bare (a young run); a past-ring gap without archive reach refuses typed.

**Catch-up extraction is fence-INCLUSIVE (defect 20, c15m).** A `round = 0` checkpoint
pointer is ambiguous: the boot snapshot (nothing folded — the guest's next expected round IS
0) and a post-round-0 capture encode the same fence. Extraction that started strictly above
the fence starved a boot-restored guest of round 0, and its very first staged record
gap-refused (`OUTCOME_STALE_RESTORE`) — both trainers respawn-looped sub-second deaths
despite correct staging. The fence round now rides along unconditionally: a genuinely folded
fence round deduplicates against the guest's resync guard (records at or below the watermark
are skipped, never gap-refused), so inclusion is safe in both readings.

**Seat replacement owns sibling replacement (defect 19, c15m).** Three invariants keep the
co-located trainer alive across seat churn: (a) an explicit join's coalesce-vs-mint decision
happens INSIDE the bring-up guard, so a join racing a completing bring-up coalesces with the
freshly inserted instance instead of minting a superseding seat; (b) a fresh seat bring-up
that finds a registered co-trainer sibling reaps it deterministically (the entry can only
belong to a replaced owner) and respawns it under the new seat; (c) the paced-respawn lane
survives transient windows (row mid-churn, primary mid-replacement) with grown backoff and
clears only on genuine teardown (completed / left / failed_terminal / intent withdrawn).
Pre-fix, the lane was cleared mid-churn and the stale sibling blocked every respawn — a
peerless coordinator idled for hours while five seat replacements each failed to bring the
trainer back.

**Staged catch-up delivery paces on guest quiescence (defect 21, c15m).** The trainer guest
pre-fetches a record's committed payloads and holds the record OUTSIDE its round driver
until they land — but an empty-entry record (a stalled round: no commits, nothing to fetch)
dispatches into the driver immediately. Presented back-to-back, a stalled round r+k enters
the driver ahead of a still-fetching round r and trips the driver's forward-contiguity
guard: `GapRefused` → `OUTCOME_STALE_RESTORE`, respawn-looping the very rejoin the catch-up
serves (c15m: fence 0, staged 0..=7 with rounds 4..=7 stalled during the defect-19 wedge —
the guest died ~1 s into every fold). Live operation never shows the shape because round
cadence spaces records out; the staged path now reproduces that spacing — after each frame
the session waits until the guest has pulled every queued event and issued no further ops
(a two-beat settle) before presenting the next (`PumpHandle::queued_events` /
`pending_ops`). Recorded non-claim: the coordinator's live ring replay on attach delivers
the same back-to-back burst and would hit the same guard if the retained ring held stalled
rounds interleaved with fetching ones; the module-side fix (hand records to the driver
immediately and let the mint's stall ladder own missing payloads) changes the module hash
and is deferred to the compatibility-class work.

**Back-pressure is not a gap (defect 22, c15m).** After a staged catch-up the live loop
drains a minutes-long backlog through a spool the guest empties one slow fold at a time. The
undelivered front frame verdicts `Backpressure` and — because the attach rewinds its cursor
for it — every frame behind it shadows as `Gap`; aging those shadows against the 20 s gap
deadline killed the session at every backlog drain, and the respawn re-staged the same
catch-up from the same fence (no checkpoint had sealed yet): a livelock. The held-frame
retry now re-arms every clock on any back-pressured pass and ages only gaps that stand
without back-pressure — the genuine missing-frame shape the deadline was built for. The
hold bound is sized for the backlog (4096), because an overflow pop drops the oldest frames
— exactly where the round records sit — manufacturing the very gaps the hold rides out.
Convergence proof (c15m live): once a fold survived attach, the trainer sealed and published
a live checkpoint at round 8, and the next respawn staged 2 records instead of 10.

### 5.6 Completion

`--stop-rounds 48` → the run reaches its terminal state on every box; `daemon-cli vhc detail`
confirms; nodes stopped; final journal + store archives pulled to the build host.

### 5.7 Abort criteria (any → stop, archive, adjudicate; never push through)

- any det-digest mismatch (hard fail, G-2);
- a drill that does not recover within `stall_rounds_max` + one checkpoint cadence;
- more than two unexplained parks;
- round wall > 3× the calibrated timer;
- any box's node crash (worker crashes are the drill domain; a node crash is a product defect — a
  finding, not a retry).

---

## 6. Evidence, replay, closure

### 6.1 The archive (one directory per box, pulled to the build host)

Node journals (all incarnations of the run), the node vhc store, node logs, the digest transcripts,
the genesis artifacts (`envelope.cbor`, `authoring-report.txt`, `run-id.txt`), the corpus manifest,
per-box binary provenance, registry evidence (seat lease history, checkpoint pointers, run state), and
the R2 record-set objects. Assemble the replay archive per the §3.4 layout contract.

### 6.2 Replay oracle (G-5)

`cargo run -p xtask -- vhc-replay --archive <dir> --run <run-id> --json` over the §3.4 archive on the
build host: both oracle modes (input replay + sandboxed consensus re-derivation against the pinned
coordinator module), every round re-derived byte-identically, verdict machine-readable into the
ledger. **The completed run's archive must certify GREEN with closure class `terminal`** (§3.4):
the recorded outcome reproduced by the replay itself, not merely a verified prefix — a `prefix`
verdict on a completed run means the terminal segment's head never published and is a finding to
adjudicate, not a pass. Drill-produced multi-chain lineages certify through the same command (the
cross-chain kernel; seams bound, rounds globally continuous and non-equivocating). On a G-2 abort,
the same command is the localization tool: it replays each peer's journal to the divergence round
and diffs the fold inputs.

### 6.3 The results ledger

`docs/specs/vhc-fleet-ceremony-ledger.md`: the fleet table with provenance, every preflight item's
evidence, the run's digest table (round → digest, one column per box), the drill timelines, the replay
verdict, the G-1..G-6 adjudication table, and every deviation with its justification. Written as the
run happens, not after.

---

## Appendix — the pre-refactor reconstruction (why the drive model changed)

The previous WAN gate had no operator CLI because *the harness itself was the operator*: a single
build-host-resident test process owned envelope authoring, run creation, peer identity, join, digest
collection, and the churn drills, with remote peers as bare `ssh -T` stdio children running a
standalone training worker over a length-framed CBOR stdio protocol. The coordinator was the deployed
cloud object; the relay ran on a small box.

That architecture was retired deliberately by the production-wiring program. Piece by piece: bare ssh
workers became full `daemon` nodes whose own supervisor spawns and owns the worker locally;
harness-derived test keys became node-minted base identities in a keystore with per-run certified
keys; direct harness joins became owner intent over the node's public API; the ad-hoc harness
envelope became the frozen, reviewed, tested genesis; the synthetic corpus became a genesis-pinned,
chunk-addressed manifest fetched over the unified data path; harness respawn became node-side
lifecycle (auto-respawn, late-join checkpoint restore). The honest post-refactor equivalent of "one
operator drives every box over ssh" is `ssh <box> daemon-cli vhc <verb>` against each box's local node
socket — the tooling in §3.

Operational lessons inherited from the pre-refactor gate: the membership floor equals the exact
initial roster; pre-realize the Mac dev env so a slow spawn never blows the join barrier; keep the
deployed cloud object in sync with the merged branch; prefer the churn-robust node lifecycle over a
bare barrier; one build at a time with capped jobs.
