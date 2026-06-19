# Rust Substrate Evaluation: choosing the runtime for the durable session lifecycle

A decision record for the substrate beneath `daemon-host`. It answers one question: **what
runtime owns the durable, suspendable, restartable session lifecycle** that every
`daemon-core` engine (and every orchestrator-agent, which is just an engine — see
[`daemon-orchestration-synthesis.md`](../research/daemon-orchestration-synthesis.md) §4.1) needs?

Companion documents:

- [`kameo-dehytration.md`](../research/kameo-dehytration.md) — the original Kameo passivation analysis that started this.
- [`daemon-lifecycle-persistence.md`](daemon-lifecycle-persistence.md) — the lifecycle/persistence contract that this decision enables.
- [`daemon-host-spec.md`](daemon-host-spec.md) — consumes this verdict; the 7 acceptance tests below are its conformance criteria.
- [`orchestration/actor-otp-supervisors/source-audit.md`](../research/source-audit.md) — **the first-hand source-tree audit** that verifies this evaluation against the actual cloned code (this doc and `kameo-dehytration.md` were originally built from web search). It confirms every load-bearing claim below and adds three sharpenings (S1 leaks-are-bounded, S2 isolation-is-placement, S3 Coerce source-confirmed), now folded in.

---

## 1. The requirement (the "strong" definition of support)

The substrate must support the **full durable virtual-entity lifecycle**, keyed by a stable
logical `SessionId`:

```text
SessionId
  -> locate / acquire the cluster owner
  -> create exactly one incarnation
  -> hydrate durable state
  -> process messages
  -> atomically checkpoint + enqueue child work
  -> destroy the incarnation and release memory
  -> persist child completion
  -> durably wake / recreate after node or process failure
  -> reject stale incarnations with fencing
```

A framework "supports" this only if it provides the **correctness-critical** middle: durable
snapshot, durable completion/wake, single-owner activation, fencing, stale-write rejection, and
recovery after the whole process is gone. Restart-on-crash, links/monitors, and clustering are
necessary but not sufficient — they are *supervision/isolation* substrates, not *activation*
runtimes.

---

## 2. Verdict

Under that definition, **no Rust actor framework provides the complete lifecycle today.** Three
touch the hard part (Elfo, Coerce, Theater); the rest are supervision/isolation only.

**Decision: plain Tokio + a durable activation layer, with `SessionStore` as the sole authority.**
Supervise only a small set of resident infrastructure services. Treat the substrate as a
**swappable trait** so that **Elfo** can later serve as an optional per-process keyed-activation
shell, but do not let any framework own durability.

This matches the independent finding from the Symphony-port survey
([`symphony-architecture-comparison.md`](../research/symphony-architecture-comparison.md)): the serious Rust
implementations chose `tokio` + SQLite and no actor framework, because *most of the important
behavior is a database transaction and ownership protocol, not actor supervision*.

---

## 3. Candidate scorecard

| Candidate | What it genuinely provides | What remains ours | Assessment |
|-----------|----------------------------|-------------------|------------|
| **Plain Tokio** | exact control over task creation, cleanup, ownership, persistence boundaries | the semantics, written explicitly | **Best production fit for this design** |
| **Elfo** | keyed lazy activation (`MapRouter` → `Unicast(SessionId)` starts an actor on demand); `RestartPolicy::never` passivates on normal exit | durable snapshot, durable completion/wake, cluster ownership, fencing, stale-write rejection, recovery scan | **Best partial fit** — removes the local activation directory only |
| **Coerce** | sharded entity identities, persistence primitives, explicit `Passivated`/`Idle`/`Active` states; **working** remoting / consistent-hash sharding / rebalancing / cluster-singleton coordinator | a *working* passivation/reactivation path + operational correctness | **Closest on paper; not usable today** — source-confirmed at HEAD 0.8.12 (2024-02-16): the reactivation loop is **dead scaffolding** (S3). Its *distributed* layer is the reference to watch for deferred fleets-of-fleets work. |
| **Ractor + `ractor-supervisor`** | dynamic children, `Temporary` policy, backoff, restart limits, meltdown handling; removes active children on termination | logical routing, durable state/wake, leases/fencing, cluster ownership | **Good infra supervisor, not an activation runtime** — and **acceptable for the bounded resident-service tree** (the `child_failure_state` leak is irrelevant under a fixed child set, S1); real caveat is its stale `ractor 0.14.3` pin. |
| **Theater** | event-chain replay, Wasm isolation, removes stopped actors | stable logical identity, implemented restart, integrated durable replay store, fleet ownership | **Research candidate only** (`restart_actor` unimplemented; new id per spawn) |
| **Elixir/OTP (via Rustler)** | excellent supervision; temporary children auto-removed on exit | durable virtual identity, passivation (hibernation ≠ dehydration), persistent wake, sharding/leases | **Strong platform substrate, still custom activation**; only if BEAM's wider benefits justify a platform change |
| **Kameo** | per-message typing, OTP-style supervision, libp2p distribution | the entire durable lifecycle | **Rejected** — no passivation; child-spec accumulation leak; mailbox-loss bug (#335) |
| **Actix / Bastion / Lunatic / joerl / Coerce-adjacent task supervisors** | restart, shutdown, links, monitors, clustering, or process isolation | essentially the entire durable entity lifecycle | **Does not materially reduce the hard work** |

### Why the actor frameworks were rejected *for passivation*

- **Kameo** — supervision restarts immediately (`Permanent`/`Transient`/`Never`); there is no
  passivation API. We build the durable layer regardless, *and* inherit a child-spec accumulation
  leak (the supervisor retains an `ErasedChildSpec` per incarnation across normal `Transient`
  exits; source-confirmed — no `children.remove` in `kameo/src/actor/kind.rs:317-362`) and the
  mailbox-loss bug ([#335](https://github.com/tqwewe/kameo/issues/335)) — now with a concrete cause:
  the shutdown path **explicitly drains and discards** queued mailbox messages
  (`kameo/src/actor/spawn.rs:216-232`). The child-spec leak is **bounded away** in our design (fixed
  resident set, S1), but the mailbox loss makes Kameo unsuitable as a session carrier regardless.
  `kameo-persistence` is a manual postcard-file snapshot add-on and is **stale** (pins `kameo 0.17.2`
  vs the current 0.20.0). See [`kameo-dehytration.md`](../research/kameo-dehytration.md).
- **Ractor** — `ractor-supervisor` is the better *infrastructure* supervisor (dynamic children,
  `Temporary`, meltdown thresholds, and it removes children from `active_children` on exit), but
  it only knows runtime child names/cells, not `SessionId → durable owner → snapshot → inbox →
  reactivation`. Source-confirmed: `child_failure_state` entries are *not* cleared on child
  termination (`ractor-supervisor/src/dynamic.rs:314-324`). **This is harmless for our use** — the
  leak only grows under unbounded *unique* children, and the resident tree is a fixed set (S1). The
  operative caveat is the crate's **stale pin** (`ractor 0.14.3` vs the current 0.15.x). Ractor's
  native supervision is the strongest in-tree (`catch_unwind` at `ractor/ractor/src/actor.rs:828`,
  monitors), but `panic = "abort"` defeats it (`lib.rs:124-125`) — the production profile must unwind.
- **Coerce** — the only crate with the right vocabulary (sharded entities, `StartEntity` /
  `PassivateEntity` / `RemoveEntity`, `Passivated` state), but source-confirmed at HEAD 0.8.12
  (2024-02-16) the loop is **dead scaffolding** (S3): a message to a `Passivated`/`Idle` entity
  returns `ActorUnavailable` with no reactivation path (`coerce/src/sharding/shard/mod.rs:347-351`);
  `post_recovery()` never starts the passivation worker (`shard/mod.rs:170-172`) and its timer body
  is empty (`shard/passivation/mod.rs:58-62`); `on_child_stopped()` is empty (`shard/mod.rs:179`);
  recovery skips passivated entities by construction (`shard/recovery.rs:58`); `save_snapshot()` is
  dead code. Its *distributed* layer (remoting, sharding, rebalancing, cluster-singleton coordinator)
  genuinely works — so watch it for the deferred distribution work, do not build the lifecycle on it.
- **Theater** — `spawn_actor()` always mints a new `TheaterId`, `restart_actor()` returns "not
  implemented", and stop removes the incarnation. Use only if Wasm isolation + deterministic
  replay are themselves primary requirements.
- **OTP/Actix/Bastion/Lunatic/joerl** — supervision/clustering/isolation only; OTP hibernation is
  a live process with reduced memory, not durable dehydration.

---

## 4. The Elfo hybrid (the only framework that reduces code without taking durability)

Elfo's `MapRouter` is the one genuinely relevant primitive. A route returning
`Outcome::Unicast(session_id)` starts a new actor for that key when none is active; with
`RestartPolicy::never`, an actor that returns normally is passivated but can be started again by a
later message:

```text
Wake(SessionId) -> MapRouter::Unicast(SessionId) -> (no active actor) -> create incarnation -> hydrate(SessionId)
```

What Elfo eliminates: the process-local `SessionId → ActorRef` activation directory.
What Elfo does **not** eliminate (stays ours): durable completion inbox, durable wake/outbox,
session snapshots, idempotent/exactly-once completion application, partition ownership, cross-node
activation leases + fencing, recovery after the Elfo process was absent at wake time, and the
guarantee that two nodes cannot activate the same `SessionId`. The Elfo message must therefore
remain a **wake hint**, never the source of truth.

Caveat: Elfo is still pre-`0.2` (published example `0.2.0-alpha.21`). Adopt only if keyed routing,
observability, and mailbox ergonomics outweigh an alpha dependency. The substrate trait keeps this
reversible.

---

## 5. Recommended architecture

> **Plain Tokio session incarnations + a durable activation layer; supervision only for the
> stable infrastructure around them.**

```text
                         +---------------------------+
SessionId  ------------->| durable partition/owner   |
                         | router + fencing lease    |
                         +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         | active-only directory     |
                         | SessionId -> Activation    |
                         +-------------+-------------+
                                       |
                                       v
                         +---------------------------+
                         | Tokio session task        |
                         | hydrated incarnation      |
                         +-------------+-------------+
                                       |
                       checkpoint + outbox transaction
                                       |
                                       v
                            task exits; memory freed
```

Completion follows the reverse durable path:

```text
worker completion
  -> txn { insert completion-inbox record; mark session Ready; enqueue durable Wake(SessionId) }
  -> partition owner receives wake
  -> acquire / increment fencing token
  -> create task; hydrate snapshot + unapplied completions
```

The in-memory active directory holds only currently running sessions. A `tokio_util`
`TaskTracker` is the right primitive: completed tracked tasks release memory immediately rather
than retaining join results.

### What to supervise (and what not to)

Supervise the small number of resident services; **never** represent a historical session
incarnation as a permanent supervisor child:

```text
Root
├── PartitionLeaseManager
├── SessionActivator
├── CompletionConsumer
├── WakeOutboxDispatcher
├── JobOutboxDispatcher
├── RecoveryScanner
└── Metrics / health services
```

Ractor (or a task-supervision crate) is reasonable for *this* tree; neither should own the durable
session identity.

---

## 6. Required acceptance tests (the host's conformance criteria)

The supervisor strategy list is irrelevant; these tests determine whether the system actually has
the lifecycle:

1. **Churn / memory baseline** — activate and passivate ≥ 1,000,000 unique `SessionId`s; the
   active-directory and any supervisor metadata return to a stable baseline (no per-incarnation
   leak).
2. **Crash-after-every-boundary** — inject a crash before snapshot, after snapshot, after job
   outbox, before task exit, after completion insert, and before wake publication; recover
   correctly each time.
3. **Wake/completion idempotency** — deliver every wake and completion repeatedly; results are
   idempotent (`UNIQUE(session_id, epoch, job_id)`).
4. **Dual-node fencing** — concurrently activate the same `SessionId` on two nodes; only the
   highest fencing token may commit.
5. **Empty-mailbox process kill** — kill the entire process while every mailbox is empty; recover
   solely from `SessionStore` + durable queues.
6. **Ownership-transfer stale-write rejection** — pause an old owner, transfer ownership, resume
   the old owner; its writes are rejected.
7. **Lost-wake recovery** — drop a wake notification entirely; a recovery scan eventually
   activates every `Ready` session.

---

## 7. Decision summary

- **Adopt:** plain Tokio + a durable activation layer; `SessionStore` authoritative; supervise
  only resident infra services; substrate behind a swappable trait.
- **Optional:** Elfo as a local keyed-activation shell (removes the activation directory only).
- **Reject for passivation:** Kameo, Ractor-as-activation-runtime, Theater, OTP-as-shortcut,
  Actix/Bastion/Lunatic/joerl.
- **Watch:** Coerce — reconsider only once its passivated-entity path transparently reactivates
  and survives crash-injection + churn tests.
- **Defer:** the cross-node distribution mechanism (Elfo/Kameo libp2p vs message bus vs gRPC)
  until fleets-of-fleets across machines is actually needed; in-process first.
