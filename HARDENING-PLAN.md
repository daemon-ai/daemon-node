# HARDENING-PLAN — authz-f3f4 (close F3 + F4 cross-owner authz leaks)

Worktree: `/home/j/experiments/daemon-worktrees/authz-f3f4` · branch `hardening/authz-f3f4`
(off `hardening/integration`). **Phase 1 = plan only. No source touched yet. Awaiting review.**

## 0. What F3/F4 are (from the conformance deny-table)

`tests/daemon-conformance/src/node/ownership_matrix.rs` records both as explicit `KnownGap`
entries in the exhaustive, no-`_` `classify()` match:

- **F4 `DeliverySessions`** — classify() line 250-252: *"delivery_sessions is transport-keyed, not
  owner-scoped"*.
- **F4 `EventsSince`** — classify() line 269: *"events feed is node-wide, not owner-scoped"*.
- **F3 `Fleet | Unit | UnitEvents | UnitOutbound | UnitHistory`** — classify() line 273-277:
  *"fleet/unit surface is not owner-scoped (UnitId→session→owner mapping TBD)"*.

The `known_gaps_are_documented_and_not_owner_gated` test (line 891-913) drives each under a
non-owner `User` (`bob`) and asserts they do **NOT** `Forbidden` today — i.e. the leak is greppable
and tracked, not fixed. `Pause | Resume | Scale` are **out of scope** and correctly stay
`NotSessionTouching` (see §5).

### Reachability confirmed (the leak is real for a non-owner `User`)

`crates/substrate/daemon-auth/src/capability.rs:143-197`: `Role::Viewer` (⊆ `User`) grants
`FleetRead`, `ControlRead`, `SessionRead`; the operator-only set (`SessionSeeAll`,
`SessionControlAny`, `FleetWrite`, `ControlWrite`) is *not* held by `User`. The coarse Auth-2 gate
(`authz.rs:44` `required_capability`) therefore admits a non-owner `User` to:

- `Fleet | Tree | Unit | UnitEvents | UnitOutbound | UnitHistory` → `FleetRead` (authz.rs:91-96)
- `EventsSince` → `ControlRead` (authz.rs:73)
- `DeliverySessions` → `SessionRead` (authz.rs:64)

…and none of these handlers apply a *per-owner* check today, so `bob` reads `alice`'s fleet
topology / unit drill-downs / node-event stream / transport session list. This is the same class as
the Phase-1 auth4 transcript leak and the F1 approvals/checkpoints leak.

## 1. The mapping that closes F3 ("UnitId→session→owner TBD" — resolved)

A unit resolves to an owner via **`UnitId → UnitNode.session → session_meta.owner`**:

- `UnitNode.session: Option<SessionId>` (`daemon-protocol/.../*.rs`, struct `UnitNode`, field at
  line 1439-1441) — the session backing a unit.
- Delegation children **inherit the parent's owner** at creation
  (`crates/node/daemon-node/src/fleet/job_worker.rs:205-212`: *"a delegated child INHERITS the
  delegating (parent) session's owner"*), so an owned subtree resolves whole and a foreign one
  drops whole — exactly what `tree_owned` (roster.rs:174-208) already relies on for `Tree`.
- A sessionless / unknown unit → owner `None` → `owner_visible(non-operator, &None) == false`
  (roster.rs:333-342), i.e. **operator-only, fail-closed** — consistent with `tree_owned`'s
  documented "sessionless unit has no owner ⇒ operator-only".

So F3 uses the **same per-row `owner_visible` filter** the merged tracks established (roster,
checkpoints, tree), keyed on the unit's backing session's owner.

## 2. Exact leaking handlers + the gate/filter per site

All handler edits are inside `daemon-host` (the single choke point every transport funnels through —
`dispatch` and the HTTP/mux adapters all call these trait methods), so gating here covers **all
transports** with **no wire change**.

### New shared helper (add to `crates/substrate/daemon-host/src/node_api/roster.rs`, near `tree_owned`)

```rust
/// Auth 4 (F3): whether the request principal may see unit `id` — resolved
/// UnitId → UnitNode.session → session_meta.owner, then `owner_visible`. A sessionless or
/// unknown unit has no owner ⇒ operator-only (fail-closed on an unknown owner). SeeAll sees all.
pub(crate) async fn unit_owner_visible(&self, id: &UnitId) -> bool {
    let principal = current_principal();
    let owner = match &self.fleet {
        Some(fleet) => match fleet.unit(id).await {
            Some(node) => match node.session {
                Some(s) => self.store.session_meta(&s).await.and_then(|m| m.owner),
                None => None,
            },
            None => None,
        },
        None => None,
    };
    owner_visible(&principal, &owner)
}
```

### F3 — fleet/unit surface (`crates/substrate/daemon-host/src/node_api/control.rs`)

| Handler | Current file:line | Response | Gate/filter to apply | Deny shape (non-owner) |
|---|---|---|---|---|
| `fleet()` | 394-399 | `FleetReport{children,usage}` | SeeAll → unchanged. Else rebuild: for each `child` in `full.children`, resolve `fleet.unit(child).session → owner`; keep visible ids; fold visible units' usage via `UsageDelta::add` (`daemon-common/src/lib.rs:301-303`). | children exclude foreign units (empty for a pure non-owner); usage folds only visible |
| `unit(id)` | 421-426 | `Option<UnitNode>` | Fetch `node = fleet.unit(&id)`; resolve `node.session → owner`; return `Some(node)` iff `owner_visible`, else `None` (reuse the fetched node — don't double-fetch). | `None` |
| `unit_events(id,max)` | 1052-1057 | `Vec<ManageEventView>` | `if !self.unit_owner_visible(&id).await { return Vec::new(); }` then delegate. | empty |
| `unit_outbound(id,max)` | 1059-1064 | `Vec<Outbound>` (**destructive drain**) | Gate **before** draining: `if !self.unit_owner_visible(&id).await { return Vec::new(); }` then delegate (a non-owner must never consume another owner's buffer). | empty |
| `unit_history(id,cursor,max)` | 1066-1069 | `JournalPageView` | `if !self.unit_owner_visible(&id).await { return JournalPageView::default(); }` then delegate. | empty page |

`current_principal`, `owner_visible`, `daemon_auth::Capability` are already in scope in control.rs
(`use super::*`; used at 209/217/404/1108). `fleet()` short-circuits SeeAll to keep operator output
byte-identical.

### F4 — node-wide event feed + delivery sessions

| Handler | File:line | Gate/filter | Deny shape |
|---|---|---|---|
| `events_page(cursor,max)` | control.rs 8-13 | After `feed.page(...)`, drop session-bearing events not owner-visible (see filter below), via shared `scope_events_page`. `current_principal()` is bound (dispatch runs in-request). | page carries no event naming a foreign session |
| `events_subscribe(cursor)` | control.rs 15-20 | Capture `current_principal()` at subscribe time (the returned stream is polled later — same rule as `tree_subscribe`, control.rs:442-445); SeeAll → return raw stream; else wrap in `.then(scope_events_page)` (self is `Clone`). **No socket.rs edit needed** — the pump (`socket.rs:1026 pump_node_events`) already runs the returned stream and just forwards pages. | each pushed page carries no foreign-session event |
| `delivery_sessions(transport,after)` | session.rs 205-217 | Before pagination, filter `self.live.delivery_sessions(&transport)` by `owner_visible(&current_principal(), &self.store.session_meta(s).owner)` per row — mirrors `checkpoints` (control.rs:1108-1119) and `roster_scoped` (roster.rs:124-132). | session absent from the returned `WirePage` |

**Events filter (shared helper on `NodeApiImpl`, add near `events_page`):**

```rust
/// Auth 4 (F4): keep only node-events a non-SeeAll principal may see. The three session-bearing
/// variants (SessionAdvanced/SessionMetaChanged/ApprovalPending) are dropped unless the session's
/// owner is visible; payload-free node-wide pointers (RosterChanged/FleetChanged/CatalogChanged/
/// DownloadProgress/ResyncNeeded) pass — the follow-up refetch they nudge is itself owner-scoped
/// (roster_scoped / tree_owned). Cursors are preserved so the client still advances correctly.
async fn scope_events_page(&self, mut page: EventsPage, principal: &Option<Principal>) -> EventsPage {
    if principal.as_ref().is_some_and(|p| p.has(Capability::SessionSeeAll)) { return page; }
    let mut kept = Vec::with_capacity(page.events.len());
    for ev in page.events {
        let session = match &ev {
            NodeEvent::SessionAdvanced { session, .. }
            | NodeEvent::SessionMetaChanged { session, .. }
            | NodeEvent::ApprovalPending { session, .. } => Some(session.clone()),
            _ => None, // node-wide pointers pass
        };
        let visible = match session {
            Some(s) => owner_visible(principal, &self.store.session_meta(&s).await.and_then(|m| m.owner)),
            None => true,
        };
        if visible { kept.push(ev); }
    }
    page.events = kept;
    page
}
```

Fail-closed: `owner_visible(None, _) == false`, so a principal-less feed reveals no session-bearing
event (matches the rest of the auth4 posture).

The `NodeEvent` variants are enumerated at `daemon-api/src/lib.rs:3379-3439` — the three
session-bearing ones carry a `session: SessionId`; the rest are payload-free.

### Files edited (for cross-track deconfliction)

- `crates/substrate/daemon-host/src/node_api/control.rs` — F3 (fleet/unit/unit_events/unit_outbound/unit_history) + F4 (events_page/events_subscribe) + `scope_events_page`.
- `crates/substrate/daemon-host/src/node_api/session.rs` — F4 (delivery_sessions).
- `crates/substrate/daemon-host/src/node_api/roster.rs` — new `unit_owner_visible` helper.
- `tests/daemon-conformance/src/node/ownership_matrix.rs` — reclassify + samples + assert arms (§3).
- `tests/daemon-conformance/src/node/f3f4_ownership.rs` (new) — dedicated bug-repro tests (§4), plus a `mod` line in `tests/daemon-conformance/src/node/mod.rs`.

**No `socket.rs` edit, no `daemon-http` edit** (the HTTP delivery bridge already wraps
`delivery_sessions` in `RequestContext::system()` (SeeAll) at daemon-http/src/lib.rs:250-254 — a
trusted node-internal consumer discovering *its* transport's sessions; my per-row filter passes
everything for SeeAll, so the bridge is unaffected while wire-user `DeliverySessions` is scoped).

## 3. Moving F3/F4 from `KnownGap` to fully-gated in `ownership_matrix.rs` (stays no-`_`)

1. **`classify()`** — delete the three `KnownGap` arms and re-home the variants into the existing
   `OwnerGated(EmptyOrAbsent)` groups (all are infallible reads that deny by returning nothing):
   - line 250-252 `DeliverySessions { .. }` → move up beside `SessionHistory | DeliveryTargets` (line 248) as `OwnerGated(EmptyOrAbsent)`.
   - line 269 `EventsSince { .. }` → `OwnerGated(EmptyOrAbsent)` in the serve_control group.
   - line 273-277 `Fleet | Unit | UnitEvents | UnitOutbound | UnitHistory` → `OwnerGated(EmptyOrAbsent)` in the serve_fleet group (next to `Tree`).
   The match stays exhaustive with **no `_` arm**.
2. **`Coverage` enum** — remove the now-unused `KnownGap(&'static str)` variant (line 219-220).
   Under `clippy -D warnings` an un-constructed variant is a build break, so removal is mandatory
   (there are zero remaining gaps: F1 is fixed/`OwnerGated`, F2 lives inside the `Fs*` samples).
3. **`known_gap_samples`** (line 715-757) — delete; move its 7 samples into `owner_gated_samples`
   (line 421-711) each paired with `Deny::EmptyOrAbsent`, reusing the existing constructors
   (`Fleet`, `Unit{unit}`, `UnitEvents`, `UnitOutbound`, `UnitHistory`, `EventsSince`,
   `DeliverySessions` with `TransportId::new("matrix/acct")`).
4. **`assert_denied`** (line 760-810) — add EmptyOrAbsent arms for the response shapes these produce
   (dispatch.rs:176-192, 100, 143):
   - `ApiResponse::Fleet(r)` → assert `r.children.is_empty()`
   - `ApiResponse::Unit(u)` → assert `u.is_none()`
   - `ApiResponse::UnitEvents(v)` → assert `v.is_empty()`
   - `ApiResponse::Drained(v)` → assert `v.is_empty()` (UnitOutbound)
   - `ApiResponse::EventsPage(p)` → assert no event names `s` (SessionAdvanced/SessionMetaChanged/ApprovalPending for `s`)
   - `ApiResponse::DeliverySessions(p)` → assert `!p.items.iter().any(|x| x == s)`
   - (`ApiResponse::Journal` for UnitHistory is **already** handled at line 767.)
5. **`known_gaps_are_documented_and_not_owner_gated`** test (line 891-913) — delete (nothing left
   to document as a gap). Update the module doc (line 16-19 + the F3/F4 mentions at 250/269/273) to
   state F3/F4 are now owner-gated.

Net: the exhaustive deny-table now **asserts F3/F4 DENY a non-owner**, driven through the real
`daemon_api::dispatch` fan-out, with the no-`_` classifier as the anti-"Nth surface" net.

> Note on table teeth: the shared `fixture` (line 815-838) seeds an approval + checkpoint + routing
> pin but **no** fleet unit / feed event / delivery binding, so the *table* entries for F3/F4 are
> green-by-emptiness even pre-fix (same as any surface the fixture doesn't populate). The table's job
> is the exhaustive classification + deny-shape guard; the genuine RED→GREEN proof is the dedicated
> repro tests in §4 (which populate each surface for real, mirroring the `f1_*` pattern).

## 4. Bug-repro tests added FIRST (RED before, GREEN after) — new `f3f4_ownership.rs`

Each mirrors `f1_approvals_pending_is_owner_scoped` (ownership_matrix.rs:84-119): seed under a
principal, assert a non-owner sees nothing, assert owner + operator (SeeAll) still see it. Handlers
are driven directly / via `dispatch` (which bypasses only the coarse Auth-2 cap gate; the per-owner
ownership logic under test runs), each under `with_request_context(ctx(name, role), …)`.

- **`f3_fleet_unit_is_owner_scoped`** — `alice` (User) `assign`s a session and it completes
  (delegation ⇒ a fleet with an `Engine` child whose `.session` inherits owner=alice; recipe from
  `tree_roster.rs:22-104`). Locate the child unit id from `node.tree(None)` under an operator.
  - `bob`: `unit(child)` → `None`; `unit_events(child)` → empty; `unit_history(child)` → empty page;
    `fleet()` → children excludes alice's units. **(RED today: bob sees them.)**
  - `alice` + `op`: `unit(child)` → `Some`; `unit_events(child)` non-empty; `fleet()` includes it.
- **`f4_events_since_is_owner_scoped`** — `alice` submits a turn to `s` ⇒ feed carries
  `SessionAdvanced{session:s}` (feed populated by submit, per `events_transport.rs:9-28`).
  - `bob`: `events_page(0, max)` carries **no** event naming `s`. **(RED today: bob sees it.)**
  - `alice` + `op`: `events_page` includes the `s`-bearing event.
- **`f4_delivery_sessions_is_owner_scoped`** — `alice` `submit_routed(origin,…)` for a
  transport-routed origin ⇒ a live session with a `Primary` delivery target on that transport
  (recipe from `delivery_memory.rs:319-361`), stamped owner=alice.
  - `bob`: `delivery_sessions(transport)` excludes alice's session. **(RED today: bob sees it.)**
  - `alice` + `op`: `delivery_sessions(transport)` includes it.

These fail on `hardening/integration` (leak) and pass after the §2 gates land. The reclassified
table tests (`every_owner_gated_variant_denies_a_non_owner`, `owner_and_operator_are_not_denied`)
add the institutional exhaustive guard.

### Regression safety of existing suites

`tree_roster.rs`, `events_transport.rs`, `delivery_memory.rs` drive their fleet/event/delivery reads
under `as_system` (SeeAll) — my SeeAll short-circuit keeps their output byte-identical, so they stay
green.

## 5. Out of scope / residual

- **`Pause | Resume | Scale`** (control.rs:1071-1090) require `FleetWrite` (authz.rs:97), which a
  non-owner `User` does **not** hold — already operator-gated at the coarse layer, not a cross-owner
  `User` leak. They stay `NotSessionTouching`. (If per-owner *operator* scoping is ever wanted, it's
  a separate follow-on; not this track.)
- **`Tree`** is already `OwnerGated` (owner-scoped via `tree_owned`) — untouched.
- **Node-wide pointer events** (`RosterChanged`/`FleetChanged`/`CatalogChanged`/`ResyncNeeded`/
  `DownloadProgress`) intentionally still pass to any authenticated principal: they carry no foreign
  session id, and the refetch they nudge (`SessionsQuery`/`Tree`/`ModelCatalog`) is itself
  owner-scoped or non-session. The `rev`/`fleet_rev` counters they carry are non-identifying.
  Residual = a coarse "something changed on the node" signal, by design.
- **`fleet().usage`** for a non-owner folds only visible units, so it no longer sums foreign work.

## 6. Wire impact — NONE (internal authz only)

No `ApiRequest`/`ApiResponse` variant added or changed, no `NodeEvent`/`FleetReport`/`UnitNode`/
`EventsPage`/`WirePage` shape change, no `daemon-api.cddl` change. Purely handler-side filtering +
test-file edits. Per the gate I will still run `cargo test -p daemon-api --features arbitrary` to
**confirm** no wire drift (expected: unchanged/green).

## 7. Cross-track deconfliction flags (for the coordinator)

- I edit **`node_api/control.rs`** and **`node_api/session.rs`**. The sibling **codec-wire-bundle**
  track touches node_api *consumer wiring* (`Origin.sender`, fingerprint). My edits are confined to
  the fleet/unit/events/delivery **authz** bodies and touch no `Origin`/`sender`/fingerprint code,
  but `session.rs` (has `submit_routed`/`record_meta` with `Origin`) and `control.rs` (routing) are
  shared files — **flagging for merge-order deconfliction**. I own fleet/unit + events/delivery
  authz; they own wire-field plumbing.
- No overlap with **env-ban-migration** (no spawn sites touched).
- No `socket.rs` edit (avoids the auth4/ingress-bounds region already in `integration`).

## 8. Exact gate (from worktree root, after approval + tests-first)

```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings   # Phase-4 disallow-lints active
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo test -p daemon-api --features arbitrary
```

Machine-load note: if `bins/daemon/tests/host_launch.rs` fails under the concurrent Opus runs,
re-run it isolated. Known flakes NOT to chase: `detached_delegation` ×2, `process_notify` store-seam.

Do NOT merge; do NOT remove the worktree.

---
**STOP — awaiting review before implementing.**
