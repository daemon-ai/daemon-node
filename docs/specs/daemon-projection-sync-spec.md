# daemon projection synchronization — one revisioned invalidation protocol

Status: accepted design; implementation staged (see §10).

Companion to [`daemon-sync-protocol-spec.md`](daemon-sync-protocol-spec.md) (L0–L4: mux envelope,
epoch-safe session resync, the `EventsSince` feed, delta lists — all shipped), whose §5 (`node-event`
union) and §6 (delta resources) this document **generalizes and supersedes**;
[`daemon-session-unification-spec.md`](daemon-session-unification-spec.md) (the durable session
authority whose `SessionAdvanced`/MergedLog semantics are consumed as-is, never modified here); and
the authoritative wire contract
[`daemon-api.cddl`](../../crates/contracts/daemon-api/daemon-api.cddl). The client-side consumption
architecture lives in `daemon-app/docs/client-sync-architecture.md`.

This document defines how **every** piece of node-owned, client-visible state propagates to every
connected client: one revisioned invalidation event over first-class revision domains, an exhaustive
mutation census with an enforcement seam, a visibility policy applied at every delivery point, and
restart-safe recovery. It exists because the current event surface is *incomplete by accretion*:
each domain that wanted cross-client freshness paid the full cost of a new wire variant, so roughly
a dozen mutating domains shipped without one and silently diverge across clients (§1.3).

---

## 0. Conclusions up front (the load-bearing decisions)

1. **One invalidation event.** All "X changed, revision N, refetch if you care" traffic collapses
   onto a single closed-enum envelope, `ProjectionChanged { domain, scope, rev, ?origin_op }`.
   Specialized events survive only where invalidation is the wrong tool: `SessionAdvanced`
   (streaming pointer), `DownloadProgress`/`QuantizeProgress` (job progress), `TransportChanged`
   (live connection presence patch), `ResyncNeeded` (feed meta). Everything else migrates (§3).

2. **The revision domain is first-class.** A revision belongs to a
   `RevisionDomain { projection, partition }` — never to a bare projection when the projection is
   partitioned, and never to an individual key. Every event, every `Bootstrap` entry, every delta
   read (`since_rev`), every client watermark, and (later) every mutation receipt refers to the
   same domain, so a scoped refetch can never mark unrelated state fresh (§2).

3. **The consistency contract is honest.** Storage and the feed use different locks, and mutations
   also originate outside `ApiRequest` dispatch (adapters, workers, ingress). The contract is
   *persist → `note_change` → reply*, with exact effect recording (`MutationEffects`, §4.2) as the
   enforcement seam — not a fictional cross-lock atomicity, and not a durable outbox. Crash windows
   are closed by rebaseline: a **node incarnation id** plus the client's persisted
   `(cursor, incarnation)` watermark makes every restart detectable (§4.3).

4. **The census is the gate.** Every `ApiRequest` variant and every internal producer is classified
   `NonStateful | MustChange(domains) | Conditional(domains)` (§5). Dispatch asserts (debug/test)
   that a `MustChange` handler recorded an effect. A mutation without an event is a bug the build
   catches, not a review nit.

5. **Visibility is a per-projection policy enforced at every delivery point** — the events page,
   the subscribe stream, *and* `Bootstrap`. Payload-free does not mean existence-safe: a keyed
   session event, a credentials rev entry, an access-control change all reveal something. Each
   projection declares a visibility class; keyed scopes without a policy are denied by default
   (§6). Bootstrap additionally has boundedness rules: domain revs only for `All`-scope and
   bounded partitions, never per-key (§8).

---

## 1. Baseline: what exists today, and where it breaks

### 1.1 The feed is sound

`NodeEventFeed` ([`internals.rs`](../../crates/substrate/daemon-host/src/node_api/internals.rs))
provides: a monotonic ring cursor with backlog paging and live broadcast (`EventsSince` as Call or
mux stream), `ResyncNeeded` on cursor age-out or subscriber lag, backlog coalescing for selected
events, per-collection revision counters bumped by `note_*` helpers, and an atomic `Bootstrap`
(cursor + epoch + rev map) taken under one lock. Delta reads (`SessionsQuery`, `ConvList`,
`RosterList`, `PersonList`) accept `since_rev` against those same counters. None of this changes
shape; it generalizes.

### 1.2 What is ad-hoc

- **Rev bookkeeping is one field per collection** (`rev`, `fleet_rev`, `profiles_rev`,
  `agents_rev`, `catalog_rev`, `notifications_rev`, per-transport `DeltaIndex`es for
  conversations/contacts, a keyed index for persons) with one hand-written `note_*` helper each.
  Adding a domain means new fields, new helpers, a new wire variant, new codec arms, new client
  policy arms — the cost structure that produced the gaps.
- **Bootstrap keys are ad-hoc strings** (`"roster"`, `"conv:{transport}"`, …) and **omit
  `agents_rev`** entirely.
- **Coalescing is an if-chain** covering `SessionAdvanced` (per session) and global latest-wins for
  `FleetChanged`/`CatalogChanged`/`ProfilesChanged`/`AgentsChanged`; nothing else coalesces.
- **The feed epoch is process-local** (`FEED_EPOCH_SEQ`) and the only restart signal; nothing on
  disk lets a client prove it is talking to the same feed generation it persisted revs against.
- **Event authorization filters exactly three variants** (the session-bearing
  `SessionAdvanced`/`SessionMetaChanged`/`ApprovalPending`, by `owner_visible`); every other event
  passes unconditionally ([`roster.rs`](../../crates/substrate/daemon-host/src/node_api/roster.rs)
  `scope_events_page`). Bootstrap is not filtered at all.
- **`EventsSince.wait_ms` is dead wire**: shaped in `wire.rs`/CDDL, implemented nowhere in
  `daemon-host`, passed only as `None` by tests. Deleted at cutover (§10, stage 7).

### 1.3 The gaps (mutations with no event — clients diverge silently)

Confirmed by exhaustive census (§5): credentials (`CredentialSet/Remove/SetLabel`, and
`AuthComplete` writing a credential), custom providers, the OpenAI gateway, tool enablement,
telemetry/crash consent, cron CRUD, saved presence, routing/chat-binding, skills and curator,
session overlays (`SetSessionModel/Mode/Overlay` persist meta **without** `SessionMetaChanged`),
`ProfileSelect`, `ModelActivate`, **`ApprovalDecide`** (a pending prompt rendered in client B never
clears when client A decides), `ContactSetAlias`, `FingerprintRevoke`, transport stored-config
writes, handover, `Assign`/`Cancel`, and all access-control writes. Clients currently paper over a
subset with connect-time refresh storms and focus-time refetches — freshness by coincidence.

---

## 2. Revision domains

```rust
/// Closed enum. Adding a projection = one new arm here + a census row + a client policy arm.
enum ProjectionId {
    Sessions, Fleet, Profiles, Skills, Curator, Catalog, Agents, Approvals,
    Credentials, CustomProviders, Gateway, Tools, TelemetryConsent, CrashConsent,
    Cron, Presence, Routing, Transports, Conversations, Contacts, Messages,
    Persons, Notifications, Vhc, Fingerprints, AccessControl,
}

/// The unit of revisioning. `partition` is present iff the projection is partitioned.
struct RevisionDomain { projection: ProjectionId, partition: Option<String> }
```

- **Partitioned projections** (partition = transport account id): `Conversations`, `Contacts`,
  `Messages`. Each partition carries its own independent monotonic rev (this is exactly today's
  per-transport `DeltaIndex`, made uniform). Partitions are bounded by construction (one per
  configured transport account).
- **Everything else is a single-domain projection** (`partition = None`).
- **Keys are not domains.** A key (session id, conversation id, profile id, run id, …) appears
  only in an event's *scope* to narrow the refetch; the revision that advances is always the
  domain's. A client that sees `rev = N` for a domain may treat the whole domain as consistent at
  `N` only after acting on every scope it received up to `N` (or refetching the domain wholesale).
- **Membership folds into `Conversations`** (key-scoped: membership is conversation detail state);
  it does not get its own projection.
- The `Sessions` domain keeps today's semantics precisely: one revision + changed-key index shared
  by roster-level and per-session-meta changes and served by `SessionsQuery{since_rev}` — the
  proven pattern each domain follows.

## 3. The wire event

### 3.1 Shape (CDDL sketch; final shapes land in `daemon-api.cddl` at stage 2)

```cddl
node-event-projection-changed = { "ProjectionChanged": {
  "projection": projection-id,          ; closed tstr enum, §2
  ? "partition": (tstr / null),         ; present iff the projection is partitioned
  "scope": change-scope,
  "rev": uint64,                        ; the domain's revision AFTER this change
  ? "origin_op": (tstr / null),         ; provenance (rung-3 op id) when known
} }

change-scope     = scope-all / scope-key
scope-all        = "All"
scope-key        = { "Key": { "key": tstr } }   ; key semantics are per-projection (§5 census)
```

Structured fields only — no composite string keys. `partition` sits beside `projection` (it names
the *domain*); `scope` narrows *within* the domain. `Key` under a partitioned projection means
"this key, in this partition".

### 3.2 Keep / migrate table

| Current variant | Fate | Domain / scope |
|---|---|---|
| `SessionAdvanced` | **keep** (streaming pointer: epoch + head_seq) | — |
| `DownloadProgress`, `QuantizeProgress` | **keep** (job progress streams) | — |
| `TransportChanged` | **keep**, narrowed to live connection/presence patching | — |
| `ResyncNeeded` | **keep** (feed meta) | — |
| `RosterChanged{rev}` | migrate | `Sessions` / All |
| `SessionMetaChanged{session,rev}` | migrate | `Sessions` / Key(session) |
| `FleetChanged{rev}` | migrate | `Fleet` / All |
| `ProfilesChanged{rev}` | migrate | `Profiles` / All |
| `CatalogChanged{rev}` | migrate | `Catalog` / All |
| `NotificationsChanged{rev}` | migrate | `Notifications` / All |
| `PersonsChanged{rev}` | migrate | `Persons` / All or Key(person) |
| `AgentsChanged{rev}` | migrate | `Agents` / All |
| `VhcChanged{run_id,rev}` | migrate | `Vhc` / Key(run) or All |
| `ContactsChanged{transport,rev}` | migrate | `Contacts`@transport / All or Key |
| `ConversationsChanged{transport,conv,change,rev}` | migrate | `Conversations`@transport / Key(conv) |
| `MembershipChanged{…}` | migrate | `Conversations`@transport / Key(conv) |
| `MessagesChanged{transport,conv}` | migrate (gains a rev via its domain) | `Messages`@transport / Key(conv) |
| `ApprovalPending{session,request_id}` | migrate **last** — only after the client approvals vertical is complete (it carries `request_id` today) | `Approvals` / Key(session) |

Migration is dual-emission first (legacy arm + envelope, §10 stage 3), removal at cutover (stage 7).
Dual emission does **not** rely on client-side rev dedupe — `MembershipChanged`, `MessagesChanged`
and `ApprovalPending` carry no rev — so each client vertical *retires its legacy handler* when it
adopts the envelope.

## 4. Consistency contract

### 4.1 The ordering guarantee

For every successful mutation of projection-visible state, in this order:

1. **Persist** (store transaction commits, or the authoritative in-memory structure is updated
   under its own lock);
2. **`note_change(domain, scope, origin_op) -> Effect { domain, rev }`** — bumps the domain rev,
   emits `ProjectionChanged` (with backlog coalescing per §7), and returns the exact revision it
   assigned;
3. **Reply** (the wire ack, if the mutation came from dispatch).

Clients observe each domain's revisions monotonically within a feed incarnation. Convergence is
asynchronous: an event is a promise that an authoritative read issued *after* seeing `rev = N`
reflects at least revision `N`.

Explicit non-guarantees: no cross-lock atomicity between store and feed (the crash window between
1 and 2 exists and is closed by §4.3), no durable outbox, no cross-domain ordering (two domains
advance independently), no event payloads (invalidation only).

### 4.2 `MutationEffects` — exact effect recording

`note_change` records its returned `Effect` into a **task-local `MutationEffects` collector**
scoped by dispatch around each request. Dispatch then enforces the census (§5):

- `MustChange(domains)` handler → at least one recorded effect, in debug/test builds a hard
  assertion (release: log + metric). The declared domain set is advisory documentation; the
  assertion is on *any* effect, so refactors that legitimately reroute a mutation to a different
  domain fail the census test (which checks declared vs recorded), not production.
- `Conditional(domains)` → zero effects is legal (a no-op update, an `Unsupported` stub).
- `NonStateful` → recording an effect is itself an error (the census is stale).

Internal producers (adapters, workers, ingress) call the same `note_change` seam without a
collector; their coverage is asserted by conformance tests, not dispatch.

Post-handler rev *sampling* is forbidden: it races concurrent mutators and cannot distinguish
no-ops, and a receipt built from it could attribute another operation's revision. The collector is
also the (deferred) receipt source (§9).

### 4.3 Restart recovery: the incarnation id

The node generates a random **incarnation id** (UUID) at process start. It is carried in
`Bootstrap` and in the mux `Hello`. The client persists `(feed_cursor, incarnation, {domain: rev})`
together (see the client architecture doc for the atomicity rules on that write). On connect:

- same incarnation → resume from the persisted cursor (`EventsSince`), trusting persisted revs;
- different incarnation (or none persisted) → full `Bootstrap` rebaseline; persisted revs are
  meaningless (feed revs are in-memory and reset).

This replaces the process-local feed-epoch comparison as the client's restart signal (the feed
epoch remains internally as the ring generation). It closes the hole where a restarted node's
reset revs can numerically equal stale persisted values and be mistaken for freshness.

## 5. The mutation census

The registry lives in code (stage 3) as an exhaustive `match` over `ApiRequest` — a new variant
does not compile without a classification. This section is the authoritative census; scope keys
are named per projection (Sessions → session id; Conversations/Messages → conversation id;
Approvals → session id; Vhc → run id; Transports → account id; Profiles → profile id).

### 5.1 Read-only variants (`NonStateful`, listed for completeness)

Health, Stats, Telemetry, Sessions, EventsSince, VerifyingKey, ApprovalsPending, FingerprintList,
CheckpointList, SessionsQuery, SessionGet, SessionSearch, SessionRecap, SessionHistory, Subscribe,
DeliveryTargets, DeliverySessions, Fleet, Tree, Unit, UnitEvents, UnitOutbound, UnitHistory,
ModelSearch, ModelFiles, ModelDownloads, ModelCatalog, ModelRecommend, ModelQuantizes,
ModelInspect, Models, ModelCurrent, ProviderCatalog, ProviderModels, CustomProviderList,
VhcRunList, VhcRunDetail, VhcHardwareReport, VhcDiskUsage, ProfileList, ProfileGet, ProfileExport,
ProfileHistory, ProfileAt, SoulGet, SkillHistory, SkillAt, SkillGet, CuratorList, CredentialList,
AuthProviders, CronList, CronRuns, CronSuggestions, RoutingListChats, RoutingGet, TransportRooms,
TransportAdapters, TransportInstances, TransportSettings, ConvList, ConvGet, ConvCreateDetails,
ConvJoinDetails, ConvHistory, ContactGetProfile, ContactActionMenu, DirectorySearch, RosterList,
AgentCatalog, ProviderList, ToolList, CommandList, Caps, ConfigGet, GatewayGet,
TelemetryConsentGet, CrashConsentGet, PresenceList, NotificationList, PersonList, FsRoots, FsList,
FsStat, FsRead, FsSearch, FsWatchPoll, BlobGet, BlobStat, UserList, RoleList, WhoAmI, Bootstrap.

Also `NonStateful`: **Poll** (queue drain, no domain state), **FeedbackSubmit** (writes an
OTLP outbox invisible to clients), **AuthBegin/AuthStep/AuthCancel** (flow-local state polled by
the initiating client only), and the `Unsupported` trait-default stubs **ToolRegister,
ProviderRegister, ConfigSet** (reclassified when implemented — the census test pins this).

### 5.2 Mutating variants

Legend: **emits?** = current behavior. Gap = currently silent (closed at stage 1 or 4).

| Variant | Class | Domain / scope | Emits today? |
|---|---|---|---|
| Submit, SubmitRouted | MustChange | Sessions/Key | yes (`note_activity`) |
| SessionCreate | MustChange | Sessions/All | yes (`RosterChanged`) |
| SessionUpdateMeta | MustChange | Sessions/Key | yes (`SessionMetaChanged`) |
| SetSessionModel/Mode/Overlay | MustChange | Sessions/Key | **gap — stage 1** |
| Assign | MustChange | Sessions/All | **gap** |
| Cancel | Conditional | Sessions/Key | **gap** |
| RecordMeta | Conditional | Sessions/Key | partial (log append) |
| Handover | MustChange | Sessions/Key | **gap** |
| CheckpointRewind, Rewind | Conditional | Sessions/Key (+ session stream) | **gap** |
| Respond | Conditional | Approvals/Key(session) | **gap** |
| ApprovalDecide | MustChange | Approvals/Key(session) | **gap** (B's prompt never clears) |
| FingerprintRevoke | MustChange | Fingerprints/All | **gap** |
| TelemetryConsentSet | MustChange | TelemetryConsent/All | **gap** |
| CrashConsentSet | MustChange | CrashConsent/All | **gap** |
| PresenceSave/Delete/SetActive | MustChange | Presence/All | **gap** |
| Pause, Resume, Scale (fleet) | Conditional | Fleet/All | stubs today |
| ModelDownload/InstallFromUrl/Cancel/Pause/Resume | Conditional | Catalog/All on completion | progress events + `CatalogChanged` |
| ModelDelete | MustChange | Catalog/All | yes |
| ModelQuantize | Conditional | Catalog/All | progress + conditional catalog |
| ModelActivate | MustChange | Profiles/Key (mutates the profile's model binding) | **gap** |
| CustomProviderSet/Remove | MustChange | CustomProviders/All | **gap** |
| VhcJoin/Leave/Pause/Resume/SwitchModule/SetPolicy | MustChange | Vhc/Key or All | yes (service `emit_changed`) |
| VhcDiskWipe | Conditional | Vhc/All | conditional |
| ProfileCreate/Update/Delete/Clone/Import/Revert, SoulSet | MustChange | Profiles/All or Key | yes |
| ProfileSelect | MustChange | Profiles/All | **gap — stage 1** |
| SkillPut, SkillRevert | MustChange | Skills/All | **gap** |
| CuratorPin/Unpin/Archive/Restore | MustChange | Curator/All | **gap** |
| CuratorRun | Conditional | Curator/All | **gap** |
| CredentialSet/Remove/SetLabel | MustChange | Credentials/All | **gap** |
| AuthComplete | MustChange | Credentials/All (+ Conditional Agents, Profiles) | partial (`AgentsChanged` for agent creds) |
| CronCreate/Update/Delete/Trigger/Pause/AcceptSuggestion/DismissSuggestion | MustChange | Cron/All | **gap** |
| RoutingSet/BindChat/UnbindChat | MustChange | Routing/All | **gap** |
| TransportSetLabel/SetEnabled/Configure/Remove | MustChange | Transports/Key(account) | partial (`TransportChanged` presence nudges only) |
| TransportConnect/Disconnect | Conditional | Transports/Key | `TransportChanged` (live presence — stays specialized) |
| ConvCreate/Join/Leave/Delete | Conditional | Conversations@t/Key | via LifecycleSink (adapter-reported — the sink is the census producer) |
| ConvSend, FtSend, FtReceive | Conditional | Messages@t/Key(conv) | via sink |
| ConvSetTopic/Title/Description | Conditional | Conversations@t/Key | adapter-reported |
| MemberInvite/Remove/Ban/SetRole | Conditional | Conversations@t/Key | via sink |
| ContactSetAlias | MustChange | Contacts@t/All | **gap** |
| RosterAdd/Update/Remove | MustChange | Contacts@t/All | yes |
| AgentDiscover | Conditional | Agents/All | yes |
| AgentRegister/Remove | MustChange | Agents/All | yes |
| ToolSetEnabled | MustChange | Tools/All | **gap** |
| CommandInvoke | Conditional | (nested — whatever the command touches) | nested |
| GatewaySet | MustChange | Gateway/All | **gap** |
| UserCreate/Disable/SetRoles/SetPassword, SessionRevoke, ResourceGrant/Revoke | MustChange | AccessControl/All | **gap** |
| FsWrite, FsWriteFromBlob, BlobPut | NonProjection (explicit) | — | FS has its own watch/etag surface (`FsWatchPoll`, `base_revision`); out of scope by design |

### 5.3 Internal producers (non-dispatch; covered by conformance, not the dispatch assertion)

| Producer | Domain(s) | Emits today? |
|---|---|---|
| LifecycleSink adapter ingress (`membership.rs`) | Conversations@t, Contacts@t, Messages@t, + live `TransportChanged` | yes (all four) |
| Notifications/persons internal APIs (`membership.rs`) | Notifications, Persons | yes |
| Fleet bus bridge, job worker, ephemeral reaper | Fleet | yes |
| Background spawner (`background.rs`) | Sessions/All (child appears in roster) | **gap** |
| Cron worker (tick/fire → runs + seeded sessions) | Cron/All + Sessions/All | **gap** |
| Notice worker (parent turn injection) | Sessions/Key | yes (activity/log path) |
| Model download/quantize jobs | Catalog + progress streams | yes |
| Agent discovery sweep, agent-auth rejection note | Agents | yes |
| Live `MergedLog::append`, durable AttachmentHub | `SessionAdvanced`, `ApprovalPending` | yes (stay specialized / migrate last) |
| Selector/commands cache change | Sessions/Key | yes |
| VHC worker | Vhc | yes |
| ProfileOps (operator + tool) | Profiles | yes |

## 6. Visibility policy

Every projection declares a class; enforcement is an exhaustive `match ProjectionId` at **all**
delivery points — `scope_events_page`, the subscribe pump, and `Bootstrap` assembly. Default-deny:
a projection without a class does not compile.

| Class | Rule | Projections |
|---|---|---|
| Public | any authenticated principal | Fleet, Profiles, Skills, Curator, Catalog, Agents, CustomProviders, Gateway, Tools, TelemetryConsent, CrashConsent, Cron, Presence, Routing, Transports, Persons, Notifications, Contacts, Conversations, Messages, Vhc |
| OwnerScoped | Key-scoped events pass iff `owner_visible(principal, owner(key))`; All-scope rev pointers pass (the refetch is itself authorization-filtered) | Sessions, Approvals |
| CapabilityScoped | pass iff the principal holds the projection's read capability | Credentials, Fingerprints, AccessControl |

Notes: this preserves today's exact behavior for the three currently-filtered variants and today's
pass-through for the rest (the deployment posture is single-user; the Public class for
messaging-adjacent projections is revisited by the access-control track, and the class table is
the single place to tighten). The specialized events keep their current rules (`SessionAdvanced` /
`ApprovalPending` OwnerScoped; progress + `TransportChanged` + `ResyncNeeded` Public). Filtered
events still advance the client's cursor (unchanged).

## 7. Coalescing matrix

Backlog-only (live broadcast always fires), as today. Rules:

1. Two backlog entries coalesce (older dropped) iff they have the **same
   `(projection, partition, scope)`** — latest-wins is safe because events are payload-free rev
   pointers and revs are monotonic.
2. Coalescing **never widens scope**: `Key(a)` and `Key(b)` entries both survive; `All` does not
   absorb `Key` entries or vice versa. (A producer that changes many keys should emit `All`.)
3. If coalesced entries carry **different `origin_op`s, the survivor's `origin_op` is nulled** —
   provenance-based echo suppression must never suppress another client's change.
4. Specialized events keep their current rules: `SessionAdvanced` coalesces per session; progress
   events do not coalesce in the ring; `ResyncNeeded` supersedes the backlog by definition.

This strictly extends today's behavior: the four global latest-wins events map to rule 1 with
`All` scope; events that never coalesced today (`RosterChanged`, `SessionMetaChanged`, …) gain
rule-1 coalescing, which is behavior-preserving for correctness (rev pointers) and strictly
reduces backlog size. The conformance suite pins rule 2 and 3 explicitly.

## 8. Bootstrap

```cddl
bootstrap-view = { "cursor": uint64, "incarnation": tstr,
                   "revs": [* domain-rev] }
domain-rev = { "projection": projection-id, ? "partition": (tstr / null), "rev": uint64 }
```

- Assembled atomically under the feed lock (unchanged), now from the uniform domain-rev table.
- Includes **every single-domain projection** (fixing the missing `agents`) and **every existing
  partition** of the partitioned projections (bounded: one per configured transport account).
  **Never per-key revs.**
- **Visibility-filtered per §6**: entries the principal cannot read are omitted. A client treats a
  missing entry as "not readable", never as "revision 0" or "stale".
- Inactive/unknown partitions are discovered lazily via events and reads, not enumerated here.
- `incarnation` replaces the client's reliance on the in-memory feed `epoch` (§4.3); `epoch`
  remains on events pages for the ring's internal lag accounting.

## 9. Deferred: receipts and optimistic concurrency (design reserved, behavior later)

- **Receipts**: mutation responses will eventually carry the recorded `MutationEffects` so the
  initiating client gets read-your-writes without waiting for the feed. Mechanism (decided now,
  shipped later): metadata rides the **mux envelope**, not the response payload —
  `Reply`/`Item` gain `? "meta": response-metadata` with
  `response-metadata = { ? "changes": [* domain-rev] }`. This covers every response type (the
  round-trip problem with typed non-`Ok` responses), touches zero response arms, and is invisible
  to the FFI/legacy path. The CDDL shape lands at cutover (stage 7) so enabling it later is
  behavior-only; population is stage 8.
- **`origin_op` discipline**: a client may use `origin_op == its own op` to skip a *redundant
  refetch* (it already holds the post-state from its own call), but must still advance its
  observed domain rev. Never suppress rev advancement.
- **Optimistic concurrency**: census-selected editable records (custom providers, cron entries,
  profiles/skills, presence, routing pins) later gain `? "expected_rev"` on their update requests;
  a mismatch returns `Conflict` (the `FsWrite base_revision` precedent). Stage 8; not required for
  convergence.

## 10. Rollout (each stage keeps the integrated bundle green)

| Stage | Repo | Content |
|---|---|---|
| 0 | node | this spec |
| 1 | node | **bug fixes with existing variants, no wire change**: overlays → `SessionMetaChanged` (emitted after durable persistence, even if live hot-apply fails); `ProfileSelect`/`ModelActivate` → `ProfilesChanged`; regression tests |
| 2 | node(+app codec) | additive CDDL + Rust: `projection-id`, `change-scope`, `ProjectionChanged`, `bootstrap-view` domain revs + incarnation, `Hello` incarnation. Old arms retained. `just update-codec`; app decodes both |
| 3 | node | rev table keyed by `RevisionDomain` behind `note_change` + `MutationEffects`; census registry + dispatch assertion; coalescing matrix; visibility classes enforced at feed + Bootstrap; migrate existing emit sites to dual emission; two-client conformance (`tests/daemon-conformance`) |
| 4 | node | close every §5 gap via `note_change`; per-domain conformance extension |
| 5 | app | one vertical at a time: entity + mapper + FetchOp + lens + GUI/TUI; retire that vertical's legacy handler |
| 6 | app | mirror `Ingestor` owns `EventsSince` + crash-safe persisted `(cursor, incarnation, revs)`; focused-engine nudge + fleet fan-out migrate; delete `SubscriptionManager`; connect sequence reduced to genuine init; explicit per-session `head_seq`/`applied_seq` freshness |
| 7 | both | remove legacy arms + dual emission; delete `wait_ms`; land `response-metadata` shape (absent behavior); WireVersion bump; two-instance GUI journey harness |
| 8 | both | receipts population + `expected_rev` OCC (behavior-only) |

### Acceptance criteria (extend the existing gates)

- **daemon-conformance**: two clients on one node — mutate via A ⇒ B observes `ProjectionChanged`
  with a strictly advancing rev ⇒ B's refetch converges, for every §5 domain; census test (declared
  vs recorded effects, exhaustive over `ApiRequest`); coalescing rules 1–3 pinned; visibility
  filtering at feed and Bootstrap (OwnerScoped and CapabilityScoped both); reconnect gap ⇒
  `ResyncNeeded`/Bootstrap recovery; incarnation change ⇒ rebaseline; dual emission during the
  migration window.
- **system-tests**: bundled two-client journeys (settings/credentials/approval-decide propagation).
- **Existing gates unchanged**: `codec-drift`, `verify-codec`, CDDL conformance + arbitrary,
  `just lint`/`just deny`, `just e2e`.
