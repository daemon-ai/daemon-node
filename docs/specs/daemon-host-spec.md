# daemon-host — the durable substrate specification

`daemon-host` is the layer that **runs `daemon-core` engine instances durably**. It is the
translation boundary (§17 downward, the management protocol upward) and the home of the **durable
activation layer**: the Tokio-based machinery that activates, suspends, persists, and reactivates
sessions keyed by a stable `SessionId`, with `SessionStore` as the sole authority.

It implements decisions taken in:

- [`rust-substrate-evaluation.md`](rust-substrate-evaluation.md) — Tokio + a durable activation layer; no actor framework owns the lifecycle; the **7 acceptance tests** are this host's conformance criteria.
- [`daemon-lifecycle-persistence.md`](daemon-lifecycle-persistence.md) — the snapshot contract, the activation architecture, the durability invariants.
- [`daemon-supervision-spec.md`](daemon-supervision-spec.md) — the management protocol the host speaks upward and translates to §17 downward.
- [`daemon-core-spec.md`](../../crates/engine/daemon-core/docs/daemon-core-spec.md) — §7 (credentials port), §14 (`SessionStore`), §16 (processes/delegation), §17 (host protocol).

`daemon-host` is a **trait with two implementations**: an **in-process embedder** (L1/L2, tests,
single-node teams) and a **remote host** speaking §17 over the wire (distributed fleets). An
orchestrator never sees which (synthesis §3).

---

## 1. Responsibilities

1. **Durable activation layer** — the SessionId → owner → task → checkpoint → exit → wake lifecycle (§2–§4).
2. **Resident-service supervision** — supervise the small fixed set of infrastructure services, never historical session incarnations (§5).
3. **Credential authority** — the authority backing the engine's §7 `CredentialProvider` port, plus fleet-wide rate/cost governance (§6).
4. **Workspace / placement provisioning** — give each engine its isolated workspace and decide where it runs (§7).
5. **Live-resource ownership** — own OS processes, LSP sessions, sockets, and child tasks so an engine can dehydrate while they persist (§8).
6. **Protocol translation** — be the one place §17 ↔ management protocol collapses (§9).

The host does **not** decide *what work to do* or *how to classify/route/gate* it — that is the
orchestrator's job ([`daemon-orchestrator-spec.md`](daemon-orchestrator-spec.md)). The host is
mechanism; the orchestrator is policy.

---

## 2. The substrate trait (swappable)

The activation substrate sits behind a trait so the default plain-Tokio implementation can be
swapped for an Elfo-backed local activation shell without touching durability
([`rust-substrate-evaluation.md`](rust-substrate-evaluation.md) §4).

```rust
#[async_trait]
trait ActivationSubstrate: Send + Sync {
    /// Ensure exactly one live incarnation for `id` in this process, hydrated and ready.
    async fn activate(&self, id: SessionId, fence: FenceToken) -> Result<ActivationRef, SubErr>;
    /// Local passivation: drop the in-memory incarnation (durability already committed).
    async fn passivate(&self, id: &SessionId);
    /// Deliver a message to the active incarnation (creating none if absent).
    async fn deliver(&self, id: &SessionId, msg: SessionMsg) -> Result<(), SubErr>;
}
```

- **Default (`tokio` impl):** an in-process active-only directory `DashMap<SessionId, ActivationRef>`
  plus `TaskTracker`-tracked session tasks. `activate` spawns/hydrates; `passivate` removes the
  directory entry and lets the task exit.
- **Optional (`elfo` impl):** a `MapRouter` keyed by `SessionId` (`Outcome::Unicast(id)` lazily
  starts a session actor; `RestartPolicy::never` passivates on normal exit). It replaces the local
  directory only — ownership, persistence, completion recovery, and fencing stay in the host layers
  below, and Elfo messages remain wake *hints*, never durable facts.

The trait is intentionally narrow: everything correctness-critical (ownership, leases, store,
outboxes) lives in the host, not the substrate, so the substrate choice is reversible.

---

## 3. The durable activation architecture

```text
                         +---------------------------+
SessionId  ------------->| PartitionLeaseManager     |  durable partition/owner router + fencing
                         +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         | active-only directory      |  (in-memory; running sessions only)
                         | SessionId -> ActivationRef  |
                         +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         | Tokio session task         |  TaskTracker-tracked; frees on exit
                         | hydrated incarnation        |  drives one daemon-core engine via §17
                         +-------------+-------------+
                                       |
                       checkpoint_and_enqueue (1 txn)
                                       |
                                       v
                            task exits; memory freed
```

Reverse (completion → wake) path:

```text
worker completion
  -> record_completion_and_wake (1 txn): insert completion inbox (UNIQUE(session,epoch,job));
                                          mark session Ready; enqueue durable Wake(SessionId)
  -> WakeOutboxDispatcher delivers the wake to the partition owner
  -> PartitionLeaseManager acquires/increments fence
  -> SessionActivator activates: load_for_activation (snapshot + unapplied completions) -> task
```

### 3.1 Session task lifecycle

A session task is the host's driver around one engine incarnation:

1. **Hydrate** — `SessionStore::load_for_activation(id, fence)` → snapshot + unapplied completions;
   reconstruct the engine from the snapshot's `Conversation` + references; re-attach host-owned live
   resources by handle (§8); apply unapplied completions idempotently *before* running new work.
2. **Run** — translate management commands to §17 `AgentCommand`s (§9); stream §17 `AgentEvent`s up
   as `ManageEvent`s; service `HostRequest`s, escalating up as needed.
3. **Suspend** — when the engine reaches a phase boundary waiting on background work: bump `epoch`,
   `checkpoint_and_enqueue(snapshot, job)` in one transaction, then **return from the task**
   (`TaskTracker` frees the memory). Persist-before-stop is mandatory.
4. **Complete** — on `TurnFinished`/`Shutdown`, emit `ManageEvent::Finished { outcome }` and exit.

A fifth boundary, **Park** (`Step::ParkApproval`), is the §12 durable edit-approval (HITL) variant of
suspend: the engine suspended on an operator decision rather than background work. The activation
layer checkpoints the snapshot and records the parked rows via `park_approval` in one transaction,
but enqueues **no** runnable job — the session stays dormant until an operator answer wakes it (§7.1).

Background work is **never a child of the session task**; it runs in a durable worker keyed by
`JobId`, so the session can dehydrate while it runs.

---

## 4. The durable store (host-owned schema)

The host owns the durable backbone, built on the §14 `SessionStore` extensions (lifecycle doc §5).
Default backend: the existing SQLite/WAL/CBOR store; a dedicated durable queue can sit behind the
same trait.

| table / queue | purpose | key invariant |
|---|---|---|
| `session_record` | `{ session_id, epoch, status, snapshot, lease, fence }` | one row per session; snapshot is CBOR (§6) |
| `completion_inbox` | durable background-job completions | `UNIQUE(session_id, epoch, job_id)` → idempotent apply |
| `wake_outbox` | durable `Wake(SessionId)` hints | at-least-once delivery; consumer is idempotent |
| `job_outbox` | durable background-job commands | at-least-once dispatch to workers |
| `pending_approvals` | parked §12 edit-approval requests (HITL) | `UNIQUE(session_id, job_id)`; a `NULL decision` keeps the session dormant (§7.1) |
| `activation_leases` | partition ownership + monotonic fencing token | only the highest fence may commit |
| `journal_entries` | the verifiable journal's append-only entries (opaque CBOR + content hash) | `UNIQUE(stream, segment, seq)`; a monotonic `cursor` keys the non-destructive read (§5.1) |
| `journal_roots` | per-segment sealed Merkle root + signature | one row per `(stream, segment)`; the rolling hash chain |

All four cross-cutting transactions (`checkpoint_and_enqueue`, `record_completion_and_wake`,
`load_for_activation`, lease acquire/renew) are single transactions on this store.

---

## 5. Resident-service supervision (and only this)

The host supervises a **small, fixed set of resident services** — never a per-session child. This
is the single design rule that keeps memory bounded at fleet scale (acceptance test #1).

```text
Root supervisor
├── PartitionLeaseManager   — owns partitions; issues/renews fencing leases; rejects stale owners
├── SessionActivator        — drives ActivationSubstrate::activate on wake / new work
├── CompletionConsumer      — applies completion_inbox records idempotently to the right session
├── WakeOutboxDispatcher    — drains wake_outbox; delivers wake hints to partition owners
├── JobOutboxDispatcher     — drains job_outbox; hands background jobs to workers
├── RecoveryScanner         — periodically scan_resumable(); activate any Ready session whose wake was lost
└── Metrics / health        — exposes Usage/RateLimit aggregation + HealthStatus
```

- A conventional supervisor (e.g. `ractor-supervisor` or a task-supervision crate) is fine for *this*
  tree — restart/backoff/meltdown semantics apply to long-lived services, not to sessions.
- Sessions are **plain `TaskTracker` tasks**, not supervised children: their durability lives in the
  store, so "restart" for a session means "re-activate from the snapshot," handled by the activation
  layer, not a supervisor child-spec.

> **Source-audit note (S1): the framework metadata leaks do not apply here.** The
> [`source-audit.md`](../research/source-audit.md) read of the cloned trees
> confirms the Kameo child-spec accumulation (`kameo/src/links.rs`, no `children.remove` on a normal
> `Transient` exit) and the `ractor-supervisor` `child_failure_state` retention
> (`ractor-supervisor/src/dynamic.rs:314-324`). **Both only bite under unbounded *unique* supervised
> children** — exactly the shape this rule forbids. Because the supervised set is small and fixed
> (the seven services above) and sessions are never supervised children, neither leak can grow. So
> `ractor-supervisor` is an acceptable choice for this tree; the only real caveat is its **stale
> dependency pin** (it pins `ractor 0.14.3` vs the current 0.15.x), not its memory behavior. A
> hand-rolled `TaskTracker` + thin restart/backoff wrapper is the equally-valid alternative and
> avoids the pin. Note also that `panic = "abort"` defeats catch-unwind-based supervision in any of
> these crates (`ractor/ractor/src/lib.rs:124-125`), so the production profile must use unwinding.

---

## 5.1 The unified verifiable journal (durable transcript history)

The host keeps **one** hash-linked, per-segment-signed chain per *stream* that carries **both**
coarse management/lifecycle records **and** the coalesced finished **chat blocks** of a transcript.
There is no separate "audit log" and "transcript store": an auditor (or a reconnecting GUI) follows
a single ordered chain to see *who managed what* and *what was said*, end to end.

- **Keyed `(stream, segment, seq)`.** A `stream` ([`JournalStreamId`]) is any addressable agent in
  the tree — a durable session, a live interactive session, or a fleet/foreign unit — so the journal
  is decoupled from the durable `(session, epoch)` identity and **every** unit journals the same way.
  A `segment` is one **turn** (streaming paths) or one **incarnation** (the durable path).
- **What is journaled (and what is not).** The host folds the fine-grained §17 stream through a
  *block coalescer*: streaming text/reasoning deltas, usage, and rate-limit snapshots are **not**
  individually journaled. Only **finished blocks** graduate into history — an assembled assistant
  message, a tool call/result (opaque structured `detail` rides through untouched), a raised host
  request, or a coalesced opaque content block (e.g. a foreign agent's terminal stream). This is the
  signing/verification unit: we seal the *finished* record, not in-progress reasoning.
- **Sealed per turn, chained.** At each turn boundary the open segment is folded into a Gordian
  Envelope whose digest is the segment **Merkle root**, signed with the node's ed25519 key, and the
  next segment chains onto that root (a rolling chain). Any mutation to an entry, the set of entries,
  or the chain is detected by re-derivation. The durable path seals **fenced** (only the incarnation
  holding the highest lease may commit, exactly like a checkpoint); non-durable streams seal
  **unfenced** (the signature is the integrity primitive — there is no competing incarnation).
- **Two reads, two purposes.** The **live drain** (`ControlApi::unit_outbound` / `SessionApi::poll`)
  is a *destructive*, best-effort, full-fidelity delta stream for a *connected* client. The
  **history read** (`ControlApi::unit_history` / `SessionApi::session_history`) is the
  *non-destructive*, cursor-paged, **decoded + verified** durable read for *reconnect / scroll-back*
  and audit: repeated reads from the same cursor return the same page, each entry decoded to its
  typed block and stamped with its sealed segment's `verified` flag.
- **Offline verification.** The node publishes its **verifying key** (`ControlApi::verifying_key`,
  `daemon-cli verifying-key`) so an auditor can verify the sealed chain without trusting the node.
  Seeding the signer from config (`DAEMON_JOURNAL_SEED`) keeps the verifying key stable across
  restarts. Across a placement cut, a placed child journals **through the parent's authoritative
  store** (the brokered store client) and seals with the config-seeded node key, so the chain
  verifies under the node's one published key without the child ever owning the parent's store.

**Role parity.** Every engine the node binary constructs is built through the same `EngineProfile`
seam (engine tunables applied) and journals per turn under the node signer — none is a bespoke,
unjournaled loop. This holds across the in-process host engines, the fleet children, the far side of
a placement cut (`DAEMON_PLACED_CHILD`), and a transport-hosted unit (`DAEMON_TRANSPORT_SERVER`). A
placed child seals through the parent's brokered store under the config-seeded key (above); a
transport node, owning its own store, seals locally and mints its own credentials via the host's
owner broker. **One deferral:** a placed child does not yet consume the parent's *credentials* over
the cut (it keeps its embedded L1 pool). The brokering primitives already exist
(`serve_credentials` / `RemoteCredentialClient`), but wiring them needs a credential channel
alongside the store cut — sequenced for when a placed child must call a real (credentialed) provider.

The crypto lives in `daemon-telemetry`; the store persists only the opaque entry bytes, content
hashes, and 32-byte roots (it never learns the protocol or the key), keeping the DAG layering clean.

---

## 6. Credential authority

The host is the **authority** backing the engine's §7 `CredentialProvider` port. The engine holds a
handle; the host's impl owns the secrets and the governance:

- **Acquire/release with scoping** — issues short-lived `CredLease`s scoped to a `ProfileRef` +
  `CredScope`; revokes on session end or cancel.
- **Rotation / cooldown / health** — the heavy logic from §7's `credential_pool` lives here at fleet
  scope (multi-key selection, `mark_exhausted`, `mark_dead`, OAuth refresh).
- **Attenuation down supervision edges** — a delegated child's `CredScope` is intersected with its
  parent's (mirrors the §16.2 toolset intersection); least-privilege is enforced by the authority,
  not trusted to the engine.
- **Fleet rate/cost governance** — because `Usage`/`RateLimit` aggregate up the management protocol
  (supervision spec §2.2), the authority can throttle a shared provider quota across many engines and
  feed cost ceilings back into `Budget` caps.

Standalone (L1) embedding uses the engine's default embedded `credential_pool` impl; under a host,
the authority-backed impl is injected at construction.

---

## 7. Workspace / placement provisioning

The host gives each engine its **isolated workspace root** (§13/§17.3 construction parameter) and
decides **where** it runs:

- **Workspace** — per-session working directory / sandbox; the orchestrator's verifier routing
  (read-only exec env, [`daemon-orchestrator-spec.md`](daemon-orchestrator-spec.md)) is realized
  here by provisioning a read-only or copy-on-write workspace variant. Workspace state is a
  **tool-owned external resource** (lifecycle doc §1.2), not part of the snapshot.
  > **Leaf sessions now do real local work in-turn.** With the engine's in-turn ReAct loop landed
  > (daemon-core-spec §4.2), a `daemon-core` leaf/session runs the real **fs** and **shell** tools
  > (daemon-core-spec §12/§13) against a §13 `ExecutionEnvironment` *within a single turn* —
  > model→tools→model until final text — rather than a single mock pass. The host roots that env at
  > the engine's workspace via the `EngineProfile`'s exec-env builder (an in-core `LocalEnvironment`
  > enforcing workspace containment + child-env scrub; the seam stays routable to a future host-owned
  > env). The loop is fully in-process; only `Effect::Delegate` still crosses the durable suspension
  > boundary, so activation/snapshot semantics are unchanged. Real networked model I/O has since
  > landed via `GenAiProvider` (Anthropic/OpenAI streaming); a deterministic mock provider is retained
  > for tests/conformance.
- **Placement** — in-process (default; same address space) or remote (a remote host driving the
  engine over §17). Placement is a host concern invisible to the orchestrator, which only routes by
  `UnitId`.
- **Brain** — in-process `daemon-core` (the reference engine, presented as an `EngineUnit`) or a
  **foreign agent** process driven through a foreign adapter (§9.1). Both are `Engine`-leaf
  `ManagedUnit`s; which brain backs a unit is a host concern, selected at spawn time from a **launch
  profile** (`program`/`args`/`env` + a `ForeignProtocol` wire selector, mirroring `PlacementSpec`) by
  a profile-driven `ChildSpawner`. A foreign brain's adapter owns its lifecycle: the durable
  activation/snapshot path (§4) is `daemon-core`-only, so a foreign unit is relaunched from its
  profile rather than rehydrated.

> **Agent adapter vs FFI — opposite directions.** Driving a *foreign* brain (above) is **us → them**:
> a host-side adapter frames §17 to a child process. The FFI crates (`bindings/`) are **them → us**: a
> non-Rust host embedding *our* engine/node. Don't conflate them.

**The consuming surface is the tree, not §17.** A GUI/TUI/`daemon-cli` never speaks §17 to individual
agents; it drives the node's `daemon-api` `ControlApi`, which projects the orchestration tree
(`tree()`/`unit()`/`unit_events()` + lifecycle `pause`/`resume`/`scale`/`cancel`/`assign`, all routed
by `UnitId`). A single agent is a tree of one; teams and fleets-of-fleets are deeper trees presented
through the same surface. The management protocol (`ManagedUnit`) is the internal recursion; the
`daemon-api` projection is its read/drive face for consumers.

**The projection is genuinely recursive (fleets-of-fleets) — sourced from the durable session graph.**
Every orchestrator, top or nested, is a parent-linked durable engine session that delegates through
the node's **single shared job outbox** (§3.1a of [`daemon-lifecycle-persistence.md`](daemon-lifecycle-persistence.md)):
a delegation suspends the parent and enqueues a job; the one `JobOutboxDispatcher` materializes a
fresh durable child session and binds it to its parent. The node therefore re-sources `tree()` /
`unit()` / `unit_events()` directly from the `SessionStore`'s parent→children graph — `root` is the
real top session (no synthetic root), `children` come from `children_of`, `state` folds from
`SessionStatus`, `work` from the delegation binding label, and `usage` from the store's per-session
fold — so a *grandchild* (and deeper) is addressable by `UnitId` at any depth, identically in-process
and over the socket/FFI. Child ids are namespaced under their parent (`{parent}/cN`) so every node is
uniquely addressable and its depth is recoverable from the id. The `ManagedUnit::project_subtree` /
`locate_*` recursion seam is now **vestigial on the durable in-process path** (the graph already spans
every depth) and is retained only for the deferred cross-node remote-host proxy; correspondingly,
lifecycle commands that only made sense for the live in-memory fleet (`pause` / `resume` / `scale`)
are currently reported `Unsupported` for durable sessions. (This is correct for an *engine leaf* but a
known gap for an *orchestrator*: a durable operator-pause/resume hold over a delegation subtree, and a
`scale` concurrency cap over the durable child graph, are specified as a design in
[`daemon-supervision-spec.md` §7](daemon-supervision-spec.md) — not yet implemented; the engine-leaf
`Ack::Unsupported` invariant is unchanged either way.) The projection DTO
(`TreeReport`/`UnitNode`/`UnitState`/`ManageEventView`) lives in `daemon-protocol` and is re-exported
by `daemon-api`, so the management contract can carry the seam without depending on the consumer
surface and the cddl wire mirror is unchanged.

**Two per-unit views: coarse dashboard vs. transcript-fidelity drill-down.** `unit_events()` is the
coarse fleet-dashboard view — a bounded buffer of `ManageEventView`s (started / progress-line /
usage / finished / error), payload-agnostic and non-destructive, what a supervisor folds. For a
chat-transcript consumer that needs to render *any* unit's full operation stream, `unit_outbound(id,
max)` is the drill-down: a destructive drain (like the per-session `poll`) of the unit's rich §17
`Outbound` stream — the full vocabulary (text, reasoning, tool I/O with the opaque structured
`detail` envelope, `ContentDelta`, usage, errors) plus blocking host requests, carried untouched.
Every engine leaf (a `daemon-core` `AgentUnit` or a foreign agent over a cut) retains this stream in
a bounded per-unit buffer; the host routes `unit_outbound` to it by `UnitId`. This is how a single
agent *or* a delegate deep in a fleet is rendered at transcript fidelity — the rich stream is
addressable by `UnitId`, not only for a top-level interactive session. The §17 ⇄ management
projection (§4) drops the opaque envelope by design, so the dashboard stays agnostic while the
drill-down stays lossless. (Durable/queryable transcript history — reconnect, scroll-back — is out of
scope for this drain, which is live-only and best-effort.)

**One node, one composition root.** The host node is assembled in exactly one place — the
`daemon-node` crate's `assemble()` — which the `daemon` binary and the conformance harness both call.
It wires the durable substrate (store + resident services), the shared job outbox worker that seeds
parent-linked durable child sessions, the credential broker, and the live session surface from a
**single orchestrator-capable `EngineProfile`** used at every depth (a node is an orchestrator iff it
actually delegated — has children — else a leaf; an `OrchestrateTool` depth guard terminates the
recursion). So the durable top, the durable nested children, and the live session paths all share one
engine shape, provider selection, brokered credentials, and engine tunables (`daemon_core::Config`).
`daemon-node` sits *above* `daemon-host` because the fleet + orchestrate-tool glue is composition
policy; `daemon-host` itself stays free of `daemon-orchestration`.

> **Provider selection is genai-native.** The host keeps **no** cloud-provider registry: the
> `ProviderSelector` is `mock | genai | daemon_api | llama_cpp | mistral_rs`, and for `genai` the
> adapter (Anthropic/OpenAI/Gemini/Groq/DeepSeek/xAI/OpenRouter/Cohere/…) is *inferred from the model
> id* by `genai` (`GenAiProvider::for_model`), with namespaced ids (`groq::…`) forcing the adapter.
> Live model listing for the GUI picker (`ModelApi::models()`) is delegated to
> `genai::Client::all_model_names` for every adapter whose key resolves, injected into the
> provider-agnostic host through the `CloudCatalog` hook (the binary owns `genai`; the host never
> links it) with a static catalog as the no-key fallback + the pricing/context overlay. Local GGUF
> models continue to come from the `ModelManager` catalog. Legacy per-provider profile names migrate
> to `genai` via serde aliases.
>
> **`daemon_api` — the OpenRouter-clone gateway (additive, WireVersion 21).** `ProviderSelector::
> DaemonApi` is a first-class, networked (not local) selector that pins genai's **OpenAI** adapter at
> the daemon-api gateway base — `https://api.daemon.ai/api/v1/` by default, overridable per profile
> (`base_url`) or at launch (`DAEMON_BASE_URL`, e.g. a local `http://127.0.0.1:8787/api/v1/`). The
> trailing slash is load-bearing: genai resolves the request URL with a relative
> `Url::join("chat/completions")`, so the base must end in `/` to yield `…/api/v1/chat/completions`
> (a base without it drops the last segment). Model ids are OpenRouter-style `author/slug` (e.g.
> `anthropic/claude-sonnet-4-5`) and the bearer is a daemon-api key. It **never** resolves the
> Anthropic-native adapter for `claude-*` ids — the gateway speaks the OpenAI Chat-Completions wire.
> Adding the selector is additive (a new enum value + serde/CDDL rule), reflected in the wire version
> bump to **21**.

### Host provider configuration (env / TOML)

The host launch reads its provider selection from environment (overriding an optional TOML file);
`config.rs` holds the authoritative doc comments for each key. There is **no silent default** — a
host launch that does not configure a provider (or configures a networked provider without a
model/credential) **fails fast** at `NodeConfig::validate_for_host()` before any store/socket setup:

| Key | Meaning |
|---|---|
| `DAEMON_MODEL_PROVIDER` | `mock \| scripted \| genai \| daemon_api \| llama \| mistralrs`. **Required** — unset fails fast. Legacy per-family names (`openai`/`anthropic`/…) map to `genai`. |
| `DAEMON_MODEL` | The model id sent to the provider. **Required** for `genai`/`daemon_api` (empty fails fast); for local kinds it is the GGUF path / HF id. |
| `DAEMON_BASE_URL` | Provider API base-URL override. For `daemon_api` it defaults to `https://api.daemon.ai/api/v1/` (slash-normalized); unset for `genai` uses the adapter's default endpoint. |
| credential | The provider bearer. Set at launch via `DAEMON_CREDENTIAL_KEY` (seeded onto the launch profile), or provisioned later over the API via `CredentialSet` on the profile. A networked provider with no credential fails fast. |

Mock/Scripted are keyless and modelless (reachable only via explicit `DAEMON_MODEL_PROVIDER=mock`/
`scripted`). Local-inference (`llama`/`mistralrs`) tuning lives under `DAEMON_INFER_*` — see
[local-inference-spec.md](local-inference-spec.md) §7. The first-admin bootstrap env keys
(`DAEMON_ADMIN_USERNAME`/`_PASSWORD`/`_PASSWORD_FILE`) are documented in
[daemon-access-control.md](daemon-access-control.md) §9.

**Background spawn — attached, fire-and-forget self-improvement.** `daemon-core` emits
`Effect::Spawn(SpawnSpec)` (→ `HostRequestKind::Spawn`, §4.6 of the core spec) for post-turn
skill/memory review. Unlike `Delegate`, it does **not** suspend the parent and is **not** routed
through the `JobOutboxDispatcher` (a job dispatch assumes a suspended parent waiting on a
`BackgroundCompletion`). Instead a `BackgroundSpawner` (held by both the durable `CoreEngineFactory`
incarnation and the live `NodeApiImpl`) materializes the child **synchronously and out-of-band**:

- It resolves `spec.kind` against a `BackgroundProfileRegistry` (kind → constrained `EngineProfile` +
  review prompt). `skill_review` is constrained to the `skill_*` toolset; `memory_review` to the
  `mnemosyne_*` toolset; both run with a bounded `max_iterations` (16) and review nudges disabled (no
  recursion). An unknown kind is a **no-op**.
- It seeds the child conversation `FromConversation` — from the parent's live conversation when raised
  mid-turn, else from the parent's last durable snapshot (`SessionStore::peek_snapshot`) — preserving
  the parent's system prompt + history, then appends the review prompt.
- It records a **child edge** (`SessionStore::record_child_edge`, not `bind_delegation`): the child is
  *tree-visible* (`children_of` folds it in, labeled by kind, so a GUI shows the review ran) but
  closes **without waking the parent** (no delegation work row → no `BackgroundCompletion`). The spawn
  is idempotent on the namespaced child id, so a recovered/duplicate spawn returns the existing child.

So a background reviewer is *attached* for audit but *fire-and-forget* for control flow — the
engine-native realization of hermes' `agent/background_review.py` daemon-thread fork. The registry is
built in `assemble()` from the node's tools and is inert (spawn no-ops) unless skills/memory tools are
present **and** the engine's review intervals are non-zero (opt-in).

**Skills subsystem (per-profile, stable-tier index + tools).** Skills are an *agent-owned* library
resolved **per profile**, exactly like memory and the context engine — not a node-global store built
once over the launch agent. When skills are enabled the binary builds a `daemon_skills::SkillsProvider`
(`per_profile` → `<data_dir>/<id>/skills`, or the legacy `fixed` single-dir override) and hands
`daemon-node` a `SkillsResolver` closure. For each session, `resolve_effective` resolves the routed
profile's own `Arc<SkillStore>` and registers *that agent's* `skills_list` / `skill_view` /
`skill_manage` tools (allowlist-gated) plus the progressive-disclosure *index* as a `StablePromptSource`
(`SkillsPromptSource`) in the stable system-prompt tier. Role engines (orchestrator/child) and the
background `skill_review` fork run the launch agent's resolved skill tools. The index is cache-stable
(names + short descriptions only); full bodies load on demand through `skill_view` — the prompt-caching
invariant hermes preserves. Writes through `skill_manage` invalidate the store's memoized index. Each
profile's store records writes through the shared `RevisionLog` and tracks a co-located `.usage.json`
sidecar (`FileSkillUsageLog`).

**Curator (per-profile skill hygiene).** A deterministic curator keeps each agent's library lean. The
`.usage.json` sidecar records per-skill `created_by` provenance (agent / user / bundled), view/use/patch
counts, `pinned`, and a lifecycle `state`. `apply_automatic_transitions` is a pure transition table
(idle → `Stale` → `Archived`, reactivate on activity) that only touches **agent-created, unpinned**
skills (operator-authored and binary-bundled skills are protected); archiving physically moves a bundle
to `<root>/.archive/` (out of discovery + the index) with revision provenance, and `restore` brings it
back. The operator surface is the per-profile `Curator{List,Pin,Unpin,Archive,Restore,Run}` family
(`daemon-cli curator …`), wire v12.

**One-lifecycle-owner invariant.** The durable and live lifecycles are intentionally distinct: a
durable session runs its engine dormant-between-turns through the activation seam (control surface,
`assign`), while a live session keeps it resident in the §17 actor (session surface, `submit`). A
single `SessionId` must never exist as two divergent engine instances, so the node claims a session
for the first surface that touches it and rejects the other with `ApiError::Conflict` until the
session is released (`cancel`). This is a lightweight guard-rail, not a merge of the two lifecycles —
the split is load-bearing for dehydration (many dormant durable sessions cost nothing) and is kept on
purpose.

> **Source-audit note (S2): isolation is a *placement* property, not a framework "distribution"
> feature.** The intuition that "distribution gives us isolation" does **not** hold for the Rust
> actor frameworks surveyed in [`source-audit.md`](../research/source-audit.md):
> Coerce/Kameo/Ractor/Elfo "distribution" is *message transport* between shared-address-space Tokio
> tasks, where a panic mid-`Arc<Mutex<_>>` can still corrupt shared state and poison the lock —
> remoting isolates nothing. True per-unit fault isolation in the cloned tree comes from exactly two
> sources, **both of which this `place` step owns**: (a) **Wasm-per-process** (only Lunatic provides
> it, at the cost of compiling the workload to Wasm), or (b) **OS process / container / remote node**
> placement. So isolation is delivered by `Provisioner::place`, not by adopting an actor crate's
> remoting layer — which reinforces, rather than changes, this design.

```rust
#[async_trait]
trait Provisioner: Send + Sync {
    async fn workspace(&self, id: &SessionId, spec: WorkspaceSpec) -> Result<WorkspaceRoot, ProvErr>;
    async fn place(&self, id: &SessionId, spec: PlacementSpec) -> Result<Placement, ProvErr>;
    async fn reclaim(&self, id: &SessionId);
}
```

---

## 7.1 Runtime control — live model switch + edit-approval HITL

The node surface exposes two **runtime-control** capabilities a GUI/operator drives on a running
session (wire v5).

**Per-session overlay (`SetSessionOverlay`, and the `SetSessionModel`/`SetSessionMode` conveniences).**
The single per-session adjustment surface is a `SessionOverlay` (model / provider / tool allowlist /
approval mode) layered on top of the session's **bound profile** at engine construction. Unlike the
durable profile (edited via `ProfileUpdate`), the overlay is the *live tweak* — but it is **persisted
as host-level session metadata** (`SessionStore::set_session_meta`, alongside the bound profile ref),
so it is **restored on rehydration** rather than lost on restart. `SetSessionModel { session, model,
provider? }` and `SetSessionMode { session, mode }` are field-scoped writes over the same overlay.
Resolution is unified: one `resolve_effective(bound profile, overlay)` builds the `EngineProfile` for
both the live surface (`LiveSessions::ensure` reads the persisted overlay when (re)spawning the actor)
and the durable path (`CoreIncarnation::hydrate` re-resolves from the `ProfileStore` + overlay instead
of pinning the factory's fixed profile). What can be hot-applied to a resident actor is — a
model/provider override sends `ActorMsg::SetProvider` (`Engine::set_provider`, applied at the next
turn boundary so an in-flight turn's prompt cache is never invalidated) and a mode override switches
the live `ParkingHandler` policy; a tool-allowlist override takes effect at the next (re)hydration
(the live tool registry is fixed for an actor's lifetime). A per-session model override feeds
`model_current`.

**Edit-approval session modes (`SetSessionMode`/`ApprovalMode`).** Each session carries an
`ApprovalPolicy` (`Ask` | `AcceptEdits` | `AutoAllow` | `Deny`, mirroring hermes' Default / Accept-Edits
/ Don't-Ask plus an explicit deny), durable on the `Snapshot` and consulted by every gated tool
action (an fs edit, a dangerous shell command). A shared `is_sensitive_path` carve-out (`.git`/`.ssh`,
dotenv, private keys) always asks regardless of policy. Autonomous durable engines (orchestrator,
delegated children, the fleet worker, background-review) default to `AutoAllow` so a headless turn
never stalls; interactive sessions default to `Ask`, and a GUI selects the mode live.

**Durable approval HITL (`ApprovalsPending`/`ApprovalDecide`).** A live session's `Ask` parks the
prompt into the drain queue and the `ParkingHandler` answers it on `respond` (or auto-allows/denies
when the live policy permits). A **durable** session cannot block a host future across restarts, so
its `Ask` mirrors the delegation suspend/resume path:

1. the gated tool asks the host; the durable `DelegateResolver` returns `HostResponseBody::Deferred(job_id)`
   (a deterministic id per `(session, post-bump epoch, ordinal)`);
2. the engine records a `PendingApproval` on its snapshot and suspends with the `await-approval`
   payload (`Effect::AwaitDecision`), producing `Step::ParkApproval`;
3. the activation layer `park_approval`s the snapshot + rows with **no** runnable job — the session
   goes dormant;
4. an operator lists parked asks (`ApprovalsPending`) and answers one (`ApprovalDecide { session,
   request_id, allow }`); the store stamps the decision, records a completion (`allow`/`deny`), and
   enqueues a wake — one transaction, idempotent per `(session, epoch, job)`;
5. the woken engine resolves the parked decision: **allow** re-runs the tool call (`pre_approved`, so
   it skips the gate and performs the side effect); **deny** injects a tool-error result. The turn
   continues so the model sees the resolved tool result.

A parked approval survives restart (it is durable in `pending_approvals` + the suspended snapshot)
and the wake-on-decision dedupes like a delegation completion (deterministic epoch + the unique row).

---

## 7.2 Profile distributions + profile/skill version history (wire v6)

The agent edits its **own** profile and skills (the background `skill_review` curator writes through
`skill_manage`), so both artifacts are versioned with a native, append-only, **content-addressed
revision log** — never a vendored git repo. One mechanism (`daemon_common::RevisionLog`, file-backed
`FileRevisionLog` in `daemon-host`) keys history by `(kind, id)` so profiles and skills share it.

- **On disk** (under the data root, beside `profiles/` and `skills/`): `revisions/<kind>/<id>/` holds
  an append-only `index.jsonl` (one `seq, parent, hash, author, reason, ts_ms` row per revision) and a
  `blobs/<sha256>.bin` content-addressed snapshot store (identical content dedupes). The blob is the
  full snapshot — a `ProfileSpec` (CBOR) or a `SkillBundle` (the `SKILL.md` + support files).
- **Provenance is first-class.** `Author` is `Operator` (a NodeApi call) or `Agent(label)` (a tool
  write, e.g. `skill_manage`). Profile mutations (`ProfileCreate`/`ProfileUpdate`/clone/
  import/revert) record `Operator`; skill writes through the store record the agent, so the curator's
  self-edits are attributable.
- **Revert is non-destructive.** `Profile{History,At,Revert}` and `Skill{History,At,Revert}`: a revert
  re-materializes an older revision's snapshot into the live store, which records a **new head** equal
  to that revision. History only grows, so **roll-forward is just reverting to a later `seq`**.
  Binary-bundled skills are read-only (revert rejected) — the same rule the reviewer follows.
- **Distributions.** A profile exports as a self-contained `Distribution { wire_version, profile,
  skills, head_seq, source }`: the `ProfileSpec` plus the profile's **local** (non-bundled) skills.
  `credential_ref` is **kept** — it is a name, not a secret (the importer registers the key via
  `CredentialSet`). `ProfileImport` validates the wire version, applies an optional id override, and
  safely extracts skills (the same path guard `write_file` uses), seeding a fresh `imported` history.
  Bundled skills are never shipped; the importing node reconstitutes them from its own binary.

The log is durable (survives restart) and wired only on durable nodes; an ephemeral node runs without
history and the versioning ops resolve to `ApiError::Unsupported`.

---

## 8. Live-resource ownership

Per the §16.1 amendment, the host owns the **live** runtime resources so an engine can dehydrate
while they persist:

- OS processes (dev servers, watchers, builds), LSP sessions, sockets, and any background child
  tasks are **host-owned**; the engine's snapshot carries only `ProcHandle`/reference views.
- On rehydration the host re-binds the engine to its handles; `ProcEvent`s/`completion` that arrived
  while the engine was dehydrated are waiting durably and surface as a `BackgroundCompletion`
  trigger (§17.1 item 5).
- On session end/reclaim the host tears these down (the `Provisioner::reclaim` + registry teardown).

---

## 9. Protocol translation (the host's defining job)

The host is the **only** node that translates: management protocol upward, §17 downward. The host is
itself **not a managed unit** — it is the adapter/substrate. It **presents each engine it drives as a
`UnitKind::Engine` `ManagedUnit`** (supervision spec §2.4) to the supervisor above it, adapting that
engine's §17 surface to satisfy the supervision-spec §4 mapping table:

- `ManageCommand::Assign { work }` → resolve `WorkRef` to a `UserMsg` → `AgentCommand::StartTurn`.
- `Cancel`/`Snapshot`/`Shutdown` → `Interrupt`/`Snapshot`/`Shutdown`.
- `Pause`/`Resume`/`Scale` → `Ack::Unsupported` (no-op at a single conversation).
- §17 `AgentEvent`s → `ManageEvent`s (`TurnStarted`→`Started`, deltas→`Progress`, `Usage`/`RateLimit`
  pass through identically, `TurnFinished`→`Finished { outcome }`).
- §17 `HostRequest`s → `ManageRequest`s; if the host cannot answer locally (no human/policy), it
  re-raises `Escalate` up its own supervisor through the management protocol.

The adapter is total upward (every §17 message maps to a `ManageEvent`/`ManageRequest`) and partial
downward (commands an engine cannot honor are `Ack::Unsupported`). §17 is **not** re-exported as the
generic types (supervision spec §4 decision); the engine crate stays free of `daemon-supervision`.

> **Framing: the host is a *tiling* over the logical tree, not a level in it
> ([`daemon-orchestration-synthesis.md`](../research/daemon-orchestration-synthesis.md) §3.2).** Because the host
> is not a managed unit, it does not sit "above" or "below" a unit — it is the runtime that holds a
> connected region of the `ManagedUnit` tree in one address space. **Placement/isolation (§7) is a
> *cut*** in that tree: a host boundary where this translation runs over the wire instead of
> in-process. Two consequences for this section: (a) the host presents *whatever sits behind it* —
> one engine, or (via an orchestrator) a whole sub-fleet — through the **same** upward face, which is
> what makes the cut placeable anywhere; (b) the translation above is **single-faced for a leaf**
> (management upward, §17 down to one engine), but for an **orchestrator node the host is two-faced on
> the management protocol** — server upward *and* client downward to its children's hosts — since the
> orchestrator engine emits only §16 delegation over §17 and the host realizes the downward
> management-protocol client + child placement. That downward-client role is the host responsibility
> that opens a cut to children; it is the precise hinge between the logical and physical structures.

### 9.1 Foreign adapters — one seam, many wire dialects

The translation above (§17 ⇄ management) is the same for **every** engine leaf. What differs per
foreign brain is only the **bytes on the cut** — real CLI agents do not speak our CBOR §17 frames;
they speak newline-delimited JSON over stdio, in one of two incompatible dialects. So the foreign
path is factored into a single reusable driver over two orthogonal seams:

- **Transport (framing)** — how the next message is delimited. `daemon-provision`'s `CutChannel`
  carries a `Framing`: `Length` (`u32`-LE length-prefixed, our native cut) or `Lines`
  (newline-delimited, for NDJSON). `Provisioner::place_lines` returns a line-framed channel; the
  spawn logic is otherwise identical to `place`.
- **Codec** — how bytes become §17 frames: `Codec::decode(&[u8]) -> Vec<Outbound>` and
  `Codec::encode(Inbound) -> Vec<Vec<u8>>`. The generic `CodecSection17<C: Codec>` owns the single
  reader task (recv → `decode` → events to the broadcast / blocking host requests through the
  `HostRequestHandler`) and the writer for `submit`. It is an ordinary `Section17Session`, so it
  reaches the supervisor through the **same** `AgentUnit::start_journaled` factory as `daemon-core`.

This removes the previously hardcoded CBOR `decode_up`/`encode_down`: that path is now just the first
codec, `NativeCutCodec` (renamed `decode_outbound`/`encode_inbound`), over the length transport.

**Protocol matrix.** `LaunchProfile.protocol: ForeignProtocol` selects how `ProfileChildSpawner`
materializes a child; all three present up the tree as a `UnitKind::Engine` `ManagedUnit` and journal
identically (sealed per turn, keyed by `UnitId`) — only the dialect differs:

| `ForeignProtocol` | Transport | Codec / adapter | Shape | Reach |
|---|---|---|---|---|
| `NativeCut` | `Length` (CBOR) | `NativeCutCodec` (in `daemon-host`) | our placed `daemon-core` children | the native dialect |
| `StreamJson` | `Lines` (NDJSON) | `StreamJsonCodec` (in `daemon-host`) | **one-way** event envelope | Claude Code; also Amp, Cursor |
| `Acp` | `Lines` (JSON-RPC 2.0) | `AcpSession` (in `daemon-acp`, on `agent-client-protocol`) | **symmetric** (agent calls back) | ~30 ACP-registry agents, incl. the in-tree Hermes Agent |

**One-way vs symmetric is the load-bearing distinction.** `stream-json` is a pure event stream: the
agent emits `system`/`assistant`/`user`/`result` envelopes carrying Anthropic content blocks, and the
only "callback" is a permission prompt the codec turns into a §17 `HostRequest::Approval`. **ACP is
symmetric**: the agent issues JSON-RPC requests *back* into the client (`session/request_permission`,
and — when advertised — `fs/*` and terminal access), which the adapter answers through the same
`HostRequestHandler`. Because the `agent-client-protocol` crate is a scoped builder/connection runtime
with its own subprocess + stdio ownership, ACP does **not** use the `CutChannel` transport at all; its
runtime is isolated in the `daemon-acp` crate behind a `Section17Session`, driven on a dedicated task
fed by an mpsc command queue so the session outlives a single prompt. The adapter ships
**permission-first** (advertises no `fs`/terminal client capabilities); fs/terminal callbacks are a
follow-up on the unchanged seam.

Codecs are **forward-compatible**: unknown message `type`s and unknown fields are ignored, per the
vendors' documented contract, so a newer agent build never breaks the adapter. All foreign codecs are
proven by mock-agent conformance tests ([`tests/daemon-conformance`](../../tests/daemon-conformance))
that spawn a real subprocess through `ProfileChildSpawner` and assert the agent (a) maps `Assign` →
`Finished{Completed}` exactly like an engine, (b) round-trips a blocking permission request through an
answer-authority, and (c) seals a journal segment that verifies under the node signing key.

---

## 10. Conformance criteria — the 7 acceptance tests

A `daemon-host` implementation (any substrate) is correct iff it passes the seven fleet-scale tests
from [`rust-substrate-evaluation.md`](rust-substrate-evaluation.md) §6:

1. **Churn / memory baseline** — activate+passivate ≥ 1,000,000 unique `SessionId`s; active
   directory + supervisor metadata return to baseline (no per-incarnation leak).
2. **Crash-after-every-boundary** — crash before snapshot, after snapshot, after job outbox, before
   task exit, after completion insert, before wake publication; recover correctly each time.
3. **Wake/completion idempotency** — deliver every wake/completion repeatedly; `UNIQUE(session,
   epoch, job)` makes apply idempotent.
4. **Dual-node fencing** — activate the same `SessionId` on two nodes; only the highest fence commits.
5. **Empty-mailbox process kill** — kill the process with all mailboxes empty; recover solely from
   `SessionStore` + durable queues.
6. **Ownership-transfer stale-write rejection** — pause an old owner, transfer ownership, resume it;
   its writes are rejected by the fence.
7. **Lost-wake recovery** — drop a wake entirely; `RecoveryScanner` eventually activates every
   `Ready` session.

These are the host's CI gates before any fleet deployment.

---

## 11. Open decisions (flagged, not blocking)

- **Store backend** — the existing SQLite/CBOR store (§14) vs a dedicated durable queue for
  inbox/outbox; kept behind the store trait.
- **Substrate** — plain-Tokio default vs Elfo local activation shell (adopt only if keyed-routing /
  observability ergonomics justify an alpha dependency).
- **Remote transport + cross-node ownership/fencing** — in-process first; the wire form of the
  management protocol and the cross-node lease protocol are deferred detail.
- **Distribution mechanism** for fleets-of-fleets (Elfo/Kameo libp2p vs message bus vs gRPC) —
  explicitly deferred until cross-node is needed.

---

## 12. Scheduled jobs — the cron backing (I15 + Phase 2)

The node runs a **5th resident service**, the `CronScheduler` (the node's `CronWorker`), alongside the
recovery/dispatch/scan/job-worker services (§5). It never builds an engine itself: a due job is fired
onto the **durable** activation path and the existing wake-outbox dispatcher runs the turn. One shared
`CronOps` surface (durable store CRUD + schedule validation via `daemon-schedule`) backs both the
operator `cron_*` control ops and the agent-facing `cron` tool, so there is one job engine, not two.

### 12.1 The contract (`CronSpec`)

A job carries its schedule (`croner` cron expression, `@every <dur>`/ISO interval, or a one-shot ISO
timestamp), `payload` (the seed prompt), and policy/shaping fields: `enabled`/`timezone`/`repeat`/
`jitter_secs`, `overlap` (skip|allow|queue) and `catch_up` (grace|skip|always), `script`/`no_agent`
(run a workspace-contained script instead of an LLM turn), `context_from` (output chaining),
`skills` (preloaded skill bodies, wire v16), the run-shaping overrides `model`/`provider`/
`enabled_toolsets`/`workdir`, `deliver` (result routing), and `origin` (the creating chat, wire v17).
Every field past v14 is additive (`#[serde(default)]`); the store keeps the spec as opaque CBOR.

### 12.2 Firing (G3 safety)

`fire()` materializes an isolated `cron_{id}_{ts}` durable session: it seeds the engine with the
(skills- and `context_from`-prefixed) payload, stamps `SessionMeta { scheduled_job, role =
EphemeralSubagent, bound_profile = target, overlay }`, records an in-flight `CronRun`, and
`enqueue_wake`s it. On hydrate the incarnation reads `scheduled_job` and arms the next turn as
`TurnTrigger::Scheduled { job }`. A cron-fired session **never** carries the `cron`/`orchestrate`
tools (G3): it hydrates under the constrained cron base profile, and the unified resolver path is
G3-safe by construction (it builds the session registry from fs+shell+node-extras+skills only). The
`cron` tool additionally refuses every action inside a `cron_*` session (defense in depth).

### 12.3 Outcome capture, delivery, and shaping (Phase 2)

- **Outcome.** When a fired session settles, `reconcile()` reads its **real** final assistant message
  read-only from the verifiable journal (the same coalesced `TranscriptBlock::Message{Assistant}` the
  history surface serves) into `CronRun.detail` + `job.last_ok`/`last_detail`, deriving `ok` from
  whether output was produced. This is what `context_from` chains downstream and what `deliver` sends.
  A run whose entire output is the `[SILENT]` sentinel is recorded `ok` but never delivered.
- **Delivery.** `deliver` is a **post-settle push** that reuses the §5.9.3 outbound surface — there is
  no parallel cron delivery path. The directive resolves to `DeliveryTarget`s the same way a live
  submit does: `"origin"` → the job's captured `Origin::primary_target()` (v17; `None` ⇒ store-only),
  `"all"` → every live `Primary`, `"<transport>:<chat>"` → a direct target. The captured text is
  carried as one assistant `TextDelta` `SessionLogEntry` and handed to the registered `DeliverySink`
  for each target's transport (the same call the per-session pump makes). The agent `cron` tool stamps
  the creating session's `Origin` (resolved from its routing pin) at create time; CLI/operator creates
  default to none.
- **Shaping.** `model`/`provider`/`enabled_toolsets`/`workdir` are projected into a `SessionOverlay`
  written to `SessionMeta.overlay` at fire time, and applied on top of the cron base at hydrate via
  the same overlay resolver the live/durable paths use (`WorkspaceBinding::Bound` → exec root).

### 12.4 Suggestions + blueprints (consent-first)

`CronOps` seeds a consent-first suggestion catalog on first read: a starter set plus any installed
`metadata.daemon.blueprint` skill (the skill→suggestion bridge). Accepting a suggestion creates its
backing job; accepted/dismissed suggestions latch out of the pending list by `dedup_key`. No job is
ever created without an explicit accept.
