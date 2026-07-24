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
  --corpus-manifest <path/to/corpus-manifest.cbor> \
  --trusted-base <hex> --trusted-base <hex> --trusted-base <hex>   # ORDERED; first = authority
  --roster <hex> --roster <hex> --roster <hex> \
  --upgrade-authority <hex> \
  --min-peers 3 --max-peers 3 --ckpt-cadence 8 --payload-retention 64 \
  --warmup-s <n> --round-max-s <n> --witness-s <n> --cooldown-s <n> --stop-rounds <n> \
  --out <dir>
```

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

- **Consensus re-derivation (sandboxed).** The pinned coordinator module is driven inside the real
  host sandbox — consensus never runs natively — over the archived driving inputs recovered from the
  sealed record segments; every archived round record must re-derive byte-identically, and every
  committed digest must recompute from the content-addressed payloads alone.
- **Per-peer digest agreement.** Each peer's recorded per-round det-state digest is compared across
  peers; a round where any peer's digest differs from the round quorum is a disagreement, and the
  earliest such round is the first divergence.

The verdict is GREEN iff the consensus oracle agrees AND (when peer transcripts are present) every
per-round digest agrees across every reporting peer; otherwise RED, carrying the first divergence
round.

#### Archive directory layout contract

```
<archive>/
  envelope.cbor                     the frozen genesis envelope bytes (authority + the pinned
                                    coordinator module hash; its blake3 IS the run id)
  coordinator.wasm                  the genesis-pinned coordinator module (blake3 == the envelope's
                                    coordinator.wasm artifact hash)
  heads.cbor                        CBOR: [ { body: <chain-head>, sigs: [ {signer, sig} ] } ]
                                    — the attested sealed-chain heads (segment 0 .. N, contiguous)
  segments/<segment_hash_hex>.seg   the sealed record-archive segment bytes (content-addressed; the
                                    file stem is the segment's blake3 content address)
  payloads/<blake3_hex>.bin         the committed update-container payload objects, by content hash
  peers/<peerid_hex>.digests.cbor   CBOR: [ [round, <16-byte digest>] ] — that peer's per-round
                                    post-ingest det-state digests (the `detail --watch` transcript)
```

`envelope.cbor`, `coordinator.wasm`, `heads.cbor`, and `segments/` are required (the consensus
oracle); `payloads/` is required for the digest-from-payloads re-verification; `peers/` is optional
(absent → the per-peer agreement section is empty, the consensus oracle still runs).

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

### 4.4 Keystores + the trust set

On each trainer: write the `[vhc]`-enabled node config (§5.1), start the node once, stop it; run
`daemon-cli vhc identity` → the box's base-identity PeerId. Collect the three PeerIds: the ordered
trust set (build host first — it is the authority), the roster (all three), and the upgrade authority
(owner decision; default recommendation: the build host only, matching the frozen test shape). Base
secrets never leave their boxes. The genesis author key is a fresh operator key minted on the build
host and archived in the ledger (public half only).

### 4.5 Corpus (before the smokes)

Tokenize the ratified corpus (TinyStories under the TinyLlama SentencePiece tokenizer) with
`tokenize-corpus`, at `--seq-len 2048 --token-width u16`. Resolve + record the exact dataset +
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
   authority, `--min-peers 3 --max-peers 3`, `--ckpt-cadence 8 --payload-retention 64`,
   `--stop-rounds 48`, and timers from the smoke calibration (round-max ≈ 3× the slowest measured
   wall; warmup generous). Review `authoring-report.txt`; the human ratifies it before seeding.
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
```

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

### 5.5 Churn drills (G-3/G-4)

- **Hard-kill drill at ~round 12** (first checkpoint exists at 8): on the Mac, `kill -9` the
  `daemon-vhc-worker` process (NOT the node — the node is the supervision layer under test). Expected:
  the node observes the worker death, respawns, re-admits under a new incarnation, restores from the
  live checkpoint, rejoins; digests agree from the restore round onward.
- **Graceful leave/rejoin drill at ~round 24:** on Windows, `daemon-cli vhc leave <run-id>`
  (graceful). With `min = max = 3` the run parks below the floor — confirm the parked state via
  `detail` for a bounded interval (2× the calibrated round wall), then `daemon-cli vhc join <run-id>`
  → restore from checkpoint → resume. Zero digest disagreement across the seam.

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
ledger. On a G-2 abort, the same command is the localization tool: it replays each peer's journal to
the divergence round and diffs the fold inputs.

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
