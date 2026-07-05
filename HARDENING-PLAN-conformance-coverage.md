# Phase 4 — Conformance coverage + ownership no-wildcard discipline (PLAN ONLY)

Worktree: `/home/j/experiments/daemon-worktrees/conformance-coverage`
Branch: `hardening/conformance-coverage` (off `hardening/integration`)
Status: **Phase 1 — plan for review. No source touched yet.**

This is the buildable slice of the plan's `conformance-cddl` item. Explicitly OUT OF SCOPE
(deferred to the Phase 5 codec bundle): any `daemon-api.cddl` edit / new wire field — there are
**no new wire fields** in `integration` yet, so we add/modify nothing on the wire and only *prove*
zero drift with `cargo test -p daemon-api --features arbitrary`. The ingress/governor no-wildcard
work (socket.rs/remote.rs/ws.rs) belongs to the sibling ingress-governor track — untouched here.

---

## 0. Review decisions (APPROVED 2026-07-05)

- **(a) Test design — APPROVED.** Single exhaustive `classify(&ApiRequest)` (NO `_` arm →
  build-break on a new unclassified variant) + a deny driver running each owner-gated sample through
  the real `daemon_api::dispatch` under non-owner **bob** (`User`, no `SessionSeeAll`), with
  alice/operator positive controls.
- **(b) `authorize_ownership` exhaustive-arm rewrite — APPROVED.** Replace `_ => Err(Forbidden)`
  (authorized.rs:142) with explicit arms over every `SessionOwnership` variant (still fail-closed),
  so a new variant forces a compile-time decision.
- **(c) F1–F4 disposition:**
  - **F1 — FIX IN-TRACK (Cluster A).** `approvals_pending` (control.rs) + the `CheckpointList`
    path (`checkpoints`) get the read-gate; land the bob-reads-alice RED→GREEN repro first.
  - **F2 — FIX IN-TRACK (localized).** Confirmed the fix is confined to `control.rs`: one
    `require_fs_root_access(&root, control)` helper that, when `root == FsRootId::Session(sid)`,
    routes through the existing `require_session_access(sid, control)` — no `WorkspaceFs` change, no
    FS-tool-surface plumbing. Applied to `fs_list`/`fs_stat`/`fs_read`/`fs_search`/`fs_watch_after`
    (read, control=false) and `write_gated` (write, control=true); `fs_roots` filters its session
    entries by `owner_visible`. Deny shape: `Forbidden` for the content ops, empty for `fs_roots`.
  - **F3 (fleet/unit) + F4 (`EventsSince`/`DeliverySessions`) — DO NOT FIX.** Represented in
    `classify()` under an explicit `KnownGap(&'static str)` category (NOT `_`, NOT omitted); the
    deny-test asserts their CURRENT behavior so the suite is green while the residual is greppable.

**Final F2 disposition: FIXED in-track** (the check is a localized `control.rs` gate — it did not
require the cross-cutting FS refactor that would have triggered a stop-and-report).

---

## 1. What "session-touching" means and how a non-owner is denied

Two-layer authz (already built in Phases 1/3):

- **Coarse capability gate** — `authz.rs::authorize` / `required_capability`, run by the *transport*
  (socket/TCP), maps each `ApiRequest` variant → one `Capability`. Already ONE exhaustive match,
  **no `_` arm** (a new variant without a mapping is a build break). This is the model to emulate.
- **Per-resource ownership gate** — enforced *below* the `NodeApi` trait, inside `NodeApiImpl`, via
  `require_session_access(session, control)` → mints the un-forgeable `AuthorizedFor<Session>`
  witness (`authorized.rs`), or the read-enumeration predicate `owner_visible(principal, owner)`
  (`roster.rs`). A non-owner `User` (holds `SessionRead/Write`, `ControlRead`, … but **not**
  `SessionSeeAll`/`SessionControlAny`) must be denied here for another user's session.

**Deny manifests in two shapes** (both are "no leak"):

| Shape | Handlers | Denied response |
|---|---|---|
| `Forbidden` (fallible ops) | submit/poll/respond/cancel/assign/handover/record_meta/set_*/approval_decide/checkpoint_rewind/rewind/session_update_meta/subscribe(Call→log_after) | `ApiResponse::Error(ApiError::Forbidden(_))` |
| empty / `None` (infallible reads — no existence oracle) | session_history, delivery_targets, session_get, session_recap, session_search, roster (sessions/sessions_query), tree | `Journal{}`/empty vec / `SessionDetail(None)` / `SessionRecap(None)` / roster excludes the id / empty tree |

The test drives a **non-owner `User` (bob)** against a session **owned by alice** and asserts the
deny shape per variant.

---

## 2. Full `ApiRequest` variant inventory + classification

Legend: **O** = owner-gated, must deny a non-owner (covered by the table);
**N** = not session-touching (its own domain/cap gate) — classified but not deny-asserted;
**F** = **FINDING**: session-touching but currently *not* owner-scoped (see §5).

### serve_session
| Variant | Class | Gate today | Deny shape |
|---|---|---|---|
| `Submit`/`SubmitRouted*`/`SessionCreate`† | O | `require_session_access(true)` in submit/submit_from/submit_as/session_create | Forbidden |
| `Poll` | O | `require_session_access(true)` | Forbidden |
| `Respond` | O | `require_session_access(true)` | Forbidden |
| `SessionHistory` | O | `require_session_access(false)` → empty | empty Journal |
| `Subscribe` (Call→`log_after`) | O | `require_session_access(true)` | Forbidden |
| `DeliveryTargets` | O | `require_session_access(false)` → empty | empty vec |
| `Handover` | O | `require_session_access(true)` | Forbidden |
| `RecordMeta` | O | `require_session_access(true)` | Forbidden |
| `SetSessionModel`/`SetSessionMode`/`SetSessionOverlay` | O | `require_session_access(true)` (+ `require_operator` for widening) | Forbidden |
| `DeliverySessions` | **F** | transport-keyed enum, **no owner scope** | (see §5, F4) |

† `SessionCreate`/`SubmitRouted` pass on an **Absent** id (creates the caller's own session, stamped
to bob) — no cross-owner leak. They deny (`Forbidden`) when the id/route resolves to alice's
**existing** session. The table targets an existing alice-owned id, so both deny.

### serve_control
| Variant | Class | Notes |
|---|---|---|
| `Health`/`Stats`/`Telemetry`/`VerifyingKey` | N | node diagnostics |
| `Sessions`/`SessionsQuery` | O | `roster_scoped` → `owner_visible` filter; roster excludes alice's id |
| `SessionGet` | O | `require_session_access(false)` → `None` |
| `SessionSearch` | O | per-hit `owner_visible` → empty |
| `SessionRecap` | O | `owner_visible` → `None` |
| `SessionUpdateMeta` | O | `require_session_access(true)` → Forbidden |
| `ApprovalDecide` | O | `require_session_access(true)` → Forbidden |
| `CheckpointRewind` | O | `require_session_access(true)` → Forbidden |
| `Assign` | O | `require_session_access(true)` (+ `ControlWrite` cap) → Forbidden |
| `Cancel` | O | `require_session_access(true)` → Forbidden |
| `Rewind` | O | `require_session_access(true)` → Forbidden |
| `ApprovalsPending { session }` | **F1** | **no owner scope** — leaks alice's approval prompts/paths to a `User` (ControlRead) |
| `CheckpointList { session }` | **F1** | **no owner scope** — leaks alice's checkpoint metadata to a `User` (ControlRead) |
| `EventsSince` | **F4** | node-wide L3 feed, `ControlRead`, not owner-scoped |

### serve_fleet
| Variant | Class | Notes |
|---|---|---|
| `Tree` | O | `tree_owned` owner-scopes subtrees → empty for a foreign tree |
| `Fleet`/`Unit`/`UnitEvents`/`UnitOutbound`/`UnitHistory` | **F3** | `FleetRead` (even Viewer); **not** owner-scoped though units back sessions |
| `Pause`/`Resume`/`Scale` | N | `FleetWrite` (operator-tier) |

### serve_fs
| Variant | Class | Notes |
|---|---|---|
| `FsList`/`FsStat`/`FsRead`/`FsSearch`/`FsWatchPoll` | **F2** (conditional) | session-touching **iff** `root == FsRootId::Session(sid)`; no owner check |
| `FsWrite`/`FsWriteFromBlob` | **F2** (conditional) | same, write side (`FsWrite` cap, held by `User`) |
| `FsRoots` | **F2** (info) | enumerates *every* live session sandbox id to any caller |
| `BlobPut`/`BlobGet`/`BlobStat` | N | content-addressed, not session-scoped |

### serve_models / serve_profile / serve_curator / serve_auth / serve_cron / serve_routing / serve_messaging / serve_registry / serve_access
All **N** (own domain + cap gate), with two flagged sub-cases:
- `RoutingBindChat { session }` carries a session id but is `RoutingWrite` (operator-tier) — a
  non-operator is blocked at the *capability* gate on the wire; via `dispatch` it does not re-check
  session ownership. Low risk (operators cross ownership anyway). Documented, not fixed here.
- `serve_access` (`SessionRevoke`, grants, user CRUD) is `AccessAdmin` — user administration, not
  session ownership.

---

## 3. Table-driven test design (the deliverable)

New file: `tests/daemon-conformance/src/node/ownership_matrix.rs` (registered in `node/mod.rs`),
sibling to `ownership.rs`. Two pillars:

### 3a. Exhaustive classifier (the "new variant = build break" forcing function)
A single `fn classify(req: &ApiRequest) -> Coverage` with **one match over `ApiRequest`, NO `_`
arm** (mirrors `authz.rs::required_capability`). Adding any future `ApiRequest` variant fails to
compile until it is explicitly classified — the anti-"hand-picked few" guarantee.

```rust
enum Coverage {
    OwnerGated(Deny),   // must deny a non-owner; the table drives + asserts it
    NotSessionTouching, // its own cap/domain gate; not deny-asserted here
}
enum Deny { Forbidden, EmptyOrAbsent } // the two no-leak shapes from §1
```

A `fn sample(req_kind) -> ApiRequest` (or a hand-listed `Vec<ApiRequest>` paired to the classifier)
builds one concrete instance per variant, each `OwnerGated` one targeting **alice's** existing
session `s`. A `#[test]` asserts every enum variant is represented (variant-count parity), so a new
variant also can't be silently dropped from the *sample* set.

### 3b. Deny driver
For each `OwnerGated` sample, run it through the **real production routing** under a bound
non-owner principal and assert the deny shape:

```rust
with_request_context(ctx("bob", Role::User), async {
    daemon_api::dispatch(&*node, req).await   // exact request→handler fan-out
}).await
```

`dispatch` reaches the `NodeApiImpl` ownership gate (it deliberately does not run the coarse cap
gate — that is the transport's job and is already covered by `authz.rs` unit tests + `negative_auth`;
here we prove **ownership** denies a principal who *does* hold the coarse cap). Assertions:
- `Deny::Forbidden` → `matches!(resp, ApiResponse::Error(ApiError::Forbidden(_)))`.
- `Deny::EmptyOrAbsent` → the response carries **none of alice's data** (empty page / `None` /
  roster/tree excludes `s`).

Fixture: reuse `assemble_with_store()` + `ctx()` from `ownership.rs`; alice opens/owns `s` (and, for
approval/checkpoint coverage, seed a parked approval + a checkpoint on `s` so a leak would be
observable — an ungated read returns a non-empty page, failing the assertion).

Positive control: the same sample under `ctx("alice")` (owner) and `ctx("op", Operator)`
(`SeeAll`/`ControlAny`) must **succeed / see the data**, proving the gate is not "always deny".

### Why in-process `dispatch`, not the socket
`ownership_transport.rs`/`http_ownership.rs` already prove the transport wiring for the pump/HTTP
paths. This table's job is breadth over *every variant's handler*, so it drives the shared
`dispatch` fan-out directly (same pattern the suite already uses for per-variant routing), keeping
the test fast and total.

---

## 4. Exhaustive-match audit (no-wildcard discipline, scope item 2)

Audited every ownership/authorization match site named in scope:

| Site | `_` arm? | Verdict |
|---|---|---|
| `authz.rs::required_capability` | **none** | Already exhaustive; the reference model. No change. |
| `authz.rs::authorize` | none (`match RequiredAccess`) | deny-default; fine. |
| `roster.rs::owner_visible` | none (`match Option`: `None`/`Some(p) if…`/`Some(p)`) | fail-closed (`None=>false`), final arm is an equality test (deny-by-default), not a silent allow. No change. |
| `roster.rs::require_operator` | none (`match Option`) | deny-default (`Some(_)=>Forbidden`, `None=>Unauthenticated`). No change. |
| `roster.rs::note_activity` `_ => None` (l.216) | yes | Benign: extracts turn text from `AgentCommand`, **not** an authz decision. Leave. |
| **`authorized.rs::authorize_ownership` `_ => Err(Forbidden)` (l.142)** | **yes** | Fail-**closed** (deny), so NOT a silent-permit today. But it is non-exhaustive over `SessionOwnership`: a future variant (e.g. `Shared`/`Delegated`) would silently fall into deny without an explicit decision. **Recommended minimal hardening:** replace `_ => Err` with explicit `Owned(_) => Err` + `LegacyUnowned => Err`, so a new `SessionOwnership` variant forces a compile decision — matching the `required_capability` discipline. Zero behavior change today. |
| `dispatch.rs` `serve_* { _ => return None }` | yes (×13) | Routing fall-through ("not my surface"), not an authz permit; an unrouted variant hits the final `unreachable!`. Not an ownership risk. Leave (and out of this track's file scope anyway). |

**Net item-2 change:** one exhaustive-arm rewrite in `authorized.rs`. Everything else already
satisfies the discipline.

---

## 5. FINDINGS — ungated session-touching surfaces (need your disposition)

These were surfaced *because* the table is exhaustive rather than hand-picked. Each is a real
cross-owner read/write reachable by a plain `User` (who holds the coarse cap). **Awaiting your
call on scope** before any fix.

- **F1 — `ApprovalsPending{session}` + `CheckpointList{session}` (control.rs).** No owner scope; a
  `User` (ControlRead) can pass another user's session id and read its parked-approval prompts/paths
  or checkpoint metadata. **Recommend: FIX in this track** — squarely the ownership layer, minimal:
  gate each with `owner_visible`/`require_session_access(false)` → deny-as-empty page, then include
  both in the table as `OwnerGated(EmptyOrAbsent)`. This turns the finding into a genuine
  bug-reproducing test (the failing-before / passing-after pair the plan wants).

- **F2 — FS session-root (`FsList/FsStat/FsRead/FsSearch/FsWatchPoll/FsWrite/FsWriteFromBlob` with
  `FsRootId::Session`, and `FsRoots` enumeration).** No owner check on the session sandbox; a `User`
  (FsRead/FsWrite) can read/write another user's session workspace. **DECISION: FIXED in-track** —
  the fix is confined to `control.rs` (a `require_fs_root_access` helper delegating to the existing
  `require_session_access`; `fs_roots` filtered by `owner_visible`), with no `WorkspaceFs` /
  FS-tool-surface plumbing, so it stayed a localized gate.

- **F3 — fleet/unit surface (`Fleet/Unit/UnitEvents/UnitOutbound/UnitHistory`).** Not owner-scoped
  (only `tree()` is, via `tree_owned`), yet `FleetRead` is held by Viewer and units back sessions.
  Owner-scoping needs a `UnitId → session → owner` mapping. **DECISION: DEFER** — represented in
  `classify()` as `KnownGap` with a doc note; the deny-test asserts current behavior. Tracked as a
  follow-on.

- **F4 — node-wide feeds (`EventsSince`, `DeliverySessions`).** Broad `ControlRead`/`SessionRead`
  reads not per-owner scoped. **DECISION: DEFER** — `KnownGap` in `classify()` + doc note; tracked
  as a follow-on.

---

## 6. Bug-reproducing angle

- If we adopt **F1**: the table (written first) FAILS on `ApprovalsPending`/`CheckpointList`
  (leaks alice's data to bob) → the two-line owner-scope fix makes it pass. Real regression pair.
- Independent of F1: the classifier's **no-`_` match** is itself the forcing function — a future
  session-touching variant added without an ownership gate cannot be classified `OwnerGated`
  truthfully (the deny driver would fail) and cannot be silently omitted (variant-count parity
  test), so "the Nth surface ships without the guard" becomes a red conformance run.

---

## 7. Residual coverage (explicitly not closed here)

- F2/F3/F4 above (pending your scope decision).
- `RoutingBindChat{session}` ownership under `dispatch` (blocked by the wire cap gate for
  non-operators; operators legitimately cross).
- `fleet()` aggregate report owner-scoping (part of F3).
- The transport-level pump/HTTP ownership is already covered by `ownership_transport.rs` /
  `http_ownership.rs`; this track adds the *breadth-over-variants* layer, not a third transport copy.

---

## 8. Exact gate (from the worktree root, all via the devShell)

```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo test -p daemon-api --features arbitrary   # prove zero wire drift
```

Machine-load note honored: if `bins/daemon`'s `host_launch.rs` trips under the parallel run,
re-run isolated (`cargo test -p daemon --test host_launch -- --test-threads=2`); known flakes
(`detached_delegation` ×2, `process_notify` store-seam) are not chased — only new/different
signatures count.

---

## 9. Files this track will touch (once approved)

- **Add:** `tests/daemon-conformance/src/node/ownership_matrix.rs` (+ one line in `node/mod.rs`).
- **Edit (item 2):** `crates/substrate/daemon-host/src/node_api/authorized.rs` — exhaustive
  `SessionOwnership` arms (drop the `_ => Err`).
- **Edit (F1 + F2):** `crates/substrate/daemon-host/src/node_api/control.rs` — owner-scope
  `approvals_pending` + `checkpoints` (F1); `require_fs_root_access` gate on the fs handlers +
  `fs_roots` owner-filter (F2).
- **No** `daemon-api.cddl` / wire-type changes. **No** socket.rs/remote.rs/ws.rs. No merge, no
  worktree removal.

**APPROVED — implementing per §0.**
