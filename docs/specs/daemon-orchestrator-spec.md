# daemon-orchestrator — the orchestration toolset and policy specification

`daemon-orchestrator` is **not a separate binary**. It is a **role**: a `daemon-core` engine
configured with an **orchestration toolset** (the agent case — judgment, learning, novelty) *or* a
**deterministic policy** implementing the same management protocol (the mechanical case — cheap,
predictable, high-volume). Both expose the *same* upward management protocol and use the *same*
downward spawn machinery, so they are interchangeable and composable
([`daemon-orchestration-synthesis.md`](../research/daemon-orchestration-synthesis.md) §4.1).

This is **where management maturity lives** (synthesis §6/§6.1): the system supports all CMMI
maturity levels by *enriching the pipeline at supervising nodes*, not by enlarging the tree.

Depends on:

- [`daemon-supervision-spec.md`](daemon-supervision-spec.md) — the protocol an orchestrator speaks both up and down.
- [`daemon-host-spec.md`](daemon-host-spec.md) — the substrate that durably runs the orchestrator engine *and* the children it spawns.
- [`daemon-core-spec.md`](../../crates/engine/daemon-core/docs/daemon-core-spec.md) §16.2 — `Effect::Delegate` / `HostRequest::Delegate`, the spawn seam.
- [`CMMI for Agentic Fleets.md`](../research/CMMI%20for%20Agentic%20Fleets.md) — verification asymmetry, the spec→gate→verify→measure→improve pipeline.

---

## 1. The fractal node

Every node is the same building block; the toolset is what makes it a leaf or a manager.

```text
node = daemon-core (brain) + daemon-host (durable substrate) + a toolset
```

- **Leaf node** — domain tools (edit, run, search). Does the work.
- **Managing node** — *pipeline tools over its subtree*. Because a managing node is itself an
  agent, the pipeline stages are just its tools, and its children are provisioned by `daemon-host`.

Recursion is free: an orchestrator speaks the management protocol up and down, so nesting
orchestrators (fleets-of-fleets) needs no new wiring — it routes `ManageEvent`s up and
`ManageCommand`s down, aggregating as it goes.

---

## 2. The orchestration toolset

The seven-stage pipeline that turns a `daemon-core` engine into a managing node. Each stage is a
**tool** the orchestrator-agent calls (or a step a deterministic policy executes); none is baked
into the engine or the host.

| stage | tool surface | maturity rung |
|---|---|---|
| **specify / decompose** | break a `WorkRef` into sub-`WorkRef`s with acceptance criteria | L3 |
| **classify / route** | label work class, risk, required gates, **oracle type** (verification asymmetry) | L3/L4 |
| **spawn / supervise** | `Delegate` child managed-units (§16.2); track their `ManageEvent`s | L2 |
| **gate (admit)** | WIP/backpressure + budget admission before dispatch | L2/L4 |
| **verify (check)** | route outputs to a verifier whose env *cannot edit* what it judges | L3 |
| **measure** | fold child `Usage`/`RateLimit`/outcomes into metrics (first-pass yield, gate-catch) | L4 |
| **improve** | rewrite the spec/gates/tools from telemetry (poka-yoke) | L5 |

### 2.1 Tracker is a tool, never a layer

Nothing tracker- or coding-specific belongs in the daemon system. The work-source/tracker is a
**pluggable tool the orchestrator-agent is given** — `tkx` by default, but equally Linear, a file,
a queue, or nothing. Symphony's poll→dispatch→reconcile loop is *one policy* a COO-agent can run,
not the architecture.

```rust
#[async_trait]
trait Tracker: Send + Sync {     // an ordinary tool, behind the model's tool interface
    async fn ready(&self) -> Vec<WorkRef>;            // tkx: `tk ready` across scopes (DAG resolved)
    async fn claim(&self, w: &WorkRef, by: AgentId) -> Result<Lease, TrackErr>;  // tkx: atomic lease + branch
    async fn update(&self, w: &WorkRef, state: WorkState);
    async fn release(&self, lease: Lease);
}
```

Consequence: the "where does a delegated child's worktree come from / is the checker read-only"
questions are **agent decisions or tool parameters**, not hardcoded rules. The orchestrator-agent
decides whether a child shares or gets its own workspace and grants a verifier a read-only
environment (provisioned by the host, §7 of the host spec).

---

## 3. The `WorkClass` binding (the "fat workflow", generalized)

A managing node binds a **work class** to an engine configuration. This generalizes Symphony's
`WORKFLOW.md` — it is *not* tied to that format, YAML, or coding.

```rust
struct WorkClassBinding {
    class: WorkClass,                 // e.g. "bugfix", "research", "triage" — opaque labels
    profile: ProfileRef,              // §2.3 engine config the child runs under
    toolset: ToolAllowlist,           // intersected with the parent's (attenuation, §16.2)
    gates: Vec<GateId>,               // admit + output gates required for this class
    budget: Budget,                   // iteration/wall/cost caps (supervision spec §2.1)
    model: Option<ModelRef>,          // provider/model selection per class
    verifier: Option<VerifierSpec>,   // oracle type + read-only env requirement
}
```

`classify/route` maps an incoming `WorkRef` → `WorkClass` → `WorkClassBinding`, which the node uses
to build the child's `DelegationSpec` (§16.2). This is the single point where policy (what config
for what work) is expressed; it is data, swappable per node and per maturity rung.

---

## 4. Two interchangeable fillings of the role

Both expose the identical `ManagedUnit` upward and use the identical `Delegate` downward.

- **Agent orchestrator** — a `daemon-core` engine whose toolset is §2. Brings judgment, learning,
  and novelty handling; nondeterministic; costs tokens per decision. It **dehydrates/rehydrates
  like any engine** ([`daemon-lifecycle-persistence.md`](daemon-lifecycle-persistence.md)): it owns
  only its `Conversation` + child references; the live children are owned by the host, so the
  orchestrator can suspend while its fleet keeps running and rehydrate on a child's
  `BackgroundCompletion`.
- **Deterministic policy orchestrator** — a state machine (Symphony-style poll→dispatch→reconcile)
  implementing `ManagedUnit` directly. Cheap, predictable, ideal for high-volume mechanical
  dispatch. No LLM cost per routing decision.

Because both sit behind the management protocol, an agent orchestrator may have a deterministic
sub-orchestrator and vice versa. Choose per node by cost-vs-judgment. Scheduler-directed dispatch
(poll a tracker) and agent-directed dispatch (`Effect::Delegate` from a reasoning engine) ride the
**same** spawn machinery (synthesis §4).

---

## 5. Verifier routing (builder ≠ checker)

L3 requires separation of roles: the thing that checks an output must not be the thing that
produced it, and must not be able to edit it.

- **Read-only execution environment** — the verifier child is provisioned by the host with a
  read-only / copy-on-write workspace variant ([`daemon-host-spec.md`](daemon-host-spec.md) §7), so
  it physically cannot mutate what it judges.
- **Toolset intersection** — the verifier's toolset is the work class's allowlist *minus* mutating
  tools; like all delegation, it never exceeds the parent's scope (§16.2 attenuation).
- **Oracle-type routing** — `classify/route` sits *early* in the pipeline because the spec→gate→
  verify pipeline only pays off where checking is cheaper/more reliable than doing (verification
  asymmetry, [`CMMI for Agentic Fleets.md`](../research/CMMI%20for%20Agentic%20Fleets.md) §5). The node routes
  work by oracle type before dispatch, not as an afterthought.

---

## 6. Maturity by addition, heterogeneous across the tree

The pipeline stages **are** the CMMI ladder, read at a supervising node:

| rung | what the managing node runs over its subtree |
|------|----------------------------------------------|
| **L1** | nothing — a leaf, no managing node |
| **L2** | bounded delegation + budgets (run-level discipline) |
| **L3** | **specify → gate → verify** as a *defined* process (builder ≠ checker) |
| **L4** | + **measure** across the subtree (first-pass yield, gate-catch, control charts) |
| **L5** | + **improve**: a node that rewrites the spec/gates/tools from its own telemetry |

Two properties this guarantees:

1. **Maturity is a property of a subtree, set by its supervising node** — so the architecture
   supports **heterogeneous maturity**: a tightly-gated, measured sub-fleet can sit under a looser
   parent, and a mature parent can supervise an experimental L1 sub-fleet. You raise maturity by
   enriching the pipeline at the *relevant* nodes, not globally.
2. **L4/L5 subsystems are first-class from day one** (no-op at low rungs), so raising maturity is a
   configuration change at a node, never a re-architecture. Tree *size* is not maturity — a large
   tree with no node-level pipeline is "automated entropy" (execution-L4 / management-L1).

---

## 7. Metrics store + telemetry schema (L4 control charts)

Measurement is possible *by construction*: `Usage`/`RateLimit` are the same type at every level of
the management protocol (supervision spec §2.2), so a node's metrics are the fold of its children's.

```rust
struct MetricsStore { /* append-only telemetry, queryable per WorkClass / subtree / window */ }

struct TelemetryRecord {
    unit: UnitId, class: WorkClass, window: TimeWindow,
    first_pass_yield: f64,     // % outputs passing gates without rework
    gate_catch_rate: f64,      // % defects caught by gates vs escaped
    rework_rate: f64,
    usage: UsageTotals,        // folded from child ManageEvent::Usage
    outcomes: OutcomeHistogram,// EndReason distribution
}
```

- Backed by the same durable store family as `SessionStore` (host spec §4); kept behind a trait.
- Published with a **CDDL schema** (same `#[non_exhaustive]` + `wire_version` discipline as §17.2
  and the management protocol), so external dashboards consume L4 **control charts** (first-pass
  yield, gate-catch, rework trends) from a stable contract while Rust stays the source of truth.
- The **improve** stage (L5) reads this store: recurring defect classes become new `GateId`s,
  updated `WorkClassBinding`s, or new tools — edited back into the policy a node owns.

---

## 8. The concrete loop (illustrative, tracker = tkx)

Drawn with `tkx` as the tracker tool and a Symphony-style policy as the dispatch behavior — both
**defaults/illustrations, not hardcoded layers** (§2.1). The invariants are the substrate (host)
and the §17/§16.2 seam.

1. **Poll** the tracker tool → candidate `WorkRef`s (DAG/blockers resolved by the tool).
2. **Classify** → `WorkClass` → `WorkClassBinding` (the classifier may itself be a `daemon-core` call).
3. **Admit** (WIP/backpressure) → dispatch only within concurrency/token budget.
4. **Claim** via the tracker tool → atomic lease + isolation (e.g. a branch/worktree).
5. **Spawn** a child managed-unit through `Delegate` (§16.2); the host provisions workspace/creds/
   placement and drives the engine over §17.
6. **Stream** child `ManageEvent`s up; answer/escalate `ManageRequest`s.
7. **Reconcile** on `Finished { outcome }`: route to a verifier (read-only env, §5); apply gates;
   on pass advance work state, on fail retry/backoff or release the lease.
8. **Measure** (L4): fold telemetry into the metrics store.
9. **Improve** (L5): control charts → new gates/docs/tools back into policy.

The orchestrator is durable throughout: it can dehydrate between any of these steps and rehydrate on
the next child completion, because the work state lives in the tracker tool and the run tree lives in
the host's store, not in the orchestrator's memory.

---

## 9. Open decisions (flagged, not blocking)

- **Policy expression format** for `WorkClassBinding` (inline Rust config vs a data file an agent
  edits); kept generic, not tied to `WORKFLOW.md`.
- **Default deterministic policy** — whether to ship a built-in Symphony-style policy orchestrator
  as a reference filling of the role.
- **Cross-node fleets-of-fleets** distribution — deferred with the host's remote transport.
