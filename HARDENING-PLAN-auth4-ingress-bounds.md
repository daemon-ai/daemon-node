# HARDENING-PLAN — Cluster A (Auth-4 ownership uniformity) + Cluster F (ingress bounds)

Branch: `hardening/auth4-ownership` (worktree `/home/j/experiments/daemon-worktrees/auth4-ownership`, base master 37d8167).
Phase 1 deliverable: plan only — no source edits, no commit.

Guiding principle: make the unsafe form unrepresentable. The core move is flipping
`None` principal from **allow** to **deny** in the ownership layer, then giving every
legitimate embedded caller an *explicit* principal (a type, `RequestContext::internal()`),
so "no identity" can no longer silently mean "full access".

## Approved additions (coordinator round 2)

1. **Scope confirmed**: daemon-matrix, daemon-rooms, `inject_session_input` (inventory items 7–11)
   are in-scope for this track.
2. **Reserved-username enforcement (NEW).** `SYSTEM_USERNAME`/`INTERNAL_USERNAME` are only doc
   markers today — nothing stops creating a real store user named `system`/`internal`. Since
   `internal` becomes an ownership stamp, enforce reservation at the daemon-auth store
   user-creation boundary. Decision: the reserved constants + list live in **daemon-auth**
   (`capability.rs`, the lowest crate that owns identity), exported as
   `RESERVED_USERNAMES`; `request_context.rs` (daemon-host) references them instead of
   redefining, so the two can't drift. `AuthStore::create_user` rejects a reserved username
   (case-insensitive) and a reserved id (ids are server-minted 64-hex so an id collision is
   already structurally impossible — rejected anyway, belt-and-suspenders). New
   `Error::ReservedUsername`. Unit test: creating `system`/`internal` fails; a normal name succeeds.
3. **daemon-common cross-track coordination.** `MAX_FRAME_BYTES` goes in its own new
   `crates/contracts/daemon-common/src/limits.rs` (with the BlobPut-256 MiB × ~2 CBOR
   array-of-ints rationale doc comment); the `lib.rs` hunk is a minimal `pub mod limits;` +
   `pub use limits::MAX_FRAME_BYTES;` so the concurrent `EnvPolicy` merge is trivial.
4. Accepted as planned: 640 MiB coarse Phase-1 bound; `control=true` on `log_after` mirroring
   `subscribe` (noted in its doc comment); the `owner="internal"` stamping side effect for
   chat/injection-created sessions (documented here and in the commit message).

---

## 0. How identity is currently bound (the mechanism)

- Task-local `REQUEST_CONTEXT: RequestContext` in
  `crates/substrate/daemon-host/src/request_context.rs:22-24`, set **only** by
  `with_request_context` (`request_context.rs:112-117`). `current_principal()`
  (`request_context.rs:123-125`) returns `None` when no scope is active. `tokio::spawn`ed
  tasks do **not** inherit it.
- The two ownership predicates read it:
  - `owner_visible(&Option<Principal>, owner)` — `roster.rs:347-356`. Today `None => true` (line 351).
  - `require_session_access(session, control)` — `roster.rs:156-182`. Today `None => return Ok(())` (line 163).
- Auth-2 capability gate `authorize` (`authz.rs:244-257`) already fails closed on `None`
  (`Unauthenticated`), but it is only run on the **wire dispatch** path
  (`socket.rs authorize_and_dispatch:332-363`, `daemon-http api_dispatch:135-162`). In-process
  trait calls bypass `authorize` entirely and rely solely on the ownership layer — which is why
  `None => allow` is the live hole.

---

## Cluster A — every session-touching surface enforces owner-or-override

### A.1 `session.rs` — handlers missing the ownership check

File: `crates/substrate/daemon-host/src/node_api/session.rs`

- **`log_after` (167-174)** — NO ownership check. Reached by `ApiRequest::Subscribe` **as a one-shot `Call`**
  (`daemon-api/src/dispatch.rs:85-95` routes `Subscribe` → `api.log_after(...)`; the wire variant doc is
  `wire.rs:118-127`). So a non-owner `Call(Subscribe{session})` reads any session's live transcript today.
  Auth-2 for `Subscribe` is only `SessionRead` (`authz.rs:60-64`), which every `User` holds — so nothing
  downstream stops a cross-owner read.
  **Fix:** add `self.require_session_access(&session, true).await?;` as the first line, mirroring
  `subscribe` (176-180) exactly. Returns `Err(Forbidden)` on deny (the return is `Result<LogPageView,_>`,
  so no empty-page fudge needed). `control=true` is deliberately identical to `subscribe` so the `Call`
  and `Open` forms of the same `Subscribe` op enforce byte-identical access; `Operator` holds both
  `SessionSeeAll` and `SessionControlAny` (`capability.rs:146-154`), so operators/dashboards are unaffected.

- **`delivery_targets` (186-188)** — NO check. Reached by `ApiRequest::DeliveryTargets` (`SessionRead`,
  `authz.rs:63`). Leaks a session's reply-routing targets cross-owner. Non-fallible return (`Vec<...>`).
  **Fix:** `if self.require_session_access(&session, false).await.is_err() { return Vec::new(); }`
  (read-of-one → `SessionSeeAll`; empty on deny, no existence oracle). `session_get`'s internal call uses
  `self.live.delivery_targets` (`control.rs:188`) — a different (live-registry) method — so gating the
  wire wrapper doesn't touch the already-owner-checked detail path.

- **`log_epoch` (182-184)** — NO check, but **not wire-reachable** (no `ApiRequest` variant; only called
  by `pump_session_log` before `subscribe`). Non-fallible (`u64`). Defense-in-depth:
  `if self.require_session_access(&session, false).await.is_err() { return 0; }`. The pump reads it before
  `subscribe`, but only ever emits it *inside* pages that `subscribe` must first authorize, so there is no
  live leak; the guard is belt-and-suspenders.

### A.2 `socket.rs` — mux stream pump runs with no bound principal (the live vuln)

File: `crates/substrate/daemon-host/src/socket.rs`

- The `Open` handler (660-705) authorizes the request under the connection principal
  (`with_request_context(ctx, ...authorize...)`, 668) but then `spawn_stream(...)` (671) spawns a **detached
  task** that does **not** inherit the task-local. `spawn_stream` (723-748) → `pump_session_log` (754-809)
  calls `api.log_epoch(...)` (763) and `api.subscribe(...)` (764) with **no context**, so the Auth-4 check
  inside `subscribe` sees `None`. Under today's `None => allow` this streams any owner's transcript to any
  authenticated peer over the mux `Open` path (the cross-owner transcript read). `tree_subscribe`
  (`control.rs:416-499`) shows the correct pattern: it *captures* `current_principal()` at subscribe time
  (433) and bakes it into the long-lived stream, because "the returned long-lived stream is polled outside
  this request's task-local scope."
  **Fix (mirror it):** thread the connection's `Principal` (+ `AuthMethod`, `conn_id`) into `spawn_stream`
  and wrap the whole spawned task body in
  `with_request_context(RequestContext::authenticated(principal, None).with_conn_id(conn_id).with_auth_method(method), async move { ... })`.
  The `Principal`/`method` are already in scope in the `Open` arm (`ConnAuth::Authenticated { principal, method }`, 661).
  Both pumps (`pump_session_log`, `pump_node_events`) then run under the connection identity — so a `User`
  streaming a session they do not own gets `End { error: Forbidden }` from `subscribe`. For the local-trust
  mux carriers the connection principal is `system()` (bound at `serve_mux:447-451`), so local admin is
  unchanged.

### A.3 `daemon-http` — session-log routes run with no bound principal

File: `crates/adapters/daemon-http/src/lib.rs`

- **How HTTP callers authenticate:** they don't, per-user. The surface is **all-or-nothing local trust**:
  without `[api].local_trust` the whole router is `deny_all` (87-93); with it, `api_dispatch`
  (135-162) and `submit_routed_tenant` (169-203) dispatch inside `RequestContext::system()`. The
  session-log routes, however, call the API with **no context**:
  - `log_after` route (258-268) → `state.api.log_after(...)` (264).
  - `subscribe_sse`/`subscribe_ws` → `open_log` (321-327) → `state.api.subscribe(...)` (324).
  - `tree_subscribe_sse` → `open_tree` (361-367) → `state.api.tree_subscribe(...)` (364).
  - `tenant_delivery_sse` (226-253) → `serve_delivery(state.api.clone(), ...)` (233).
  After A.1/A.4 these would newly **deny** (`None` → deny) and break the routes.
  **Fix (bind the same identity `api_dispatch` uses):** wrap each route's api call in
  `with_request_context(RequestContext::system(), async { ... }.await)` (HTTP is local-trust-only, so
  `system` is the honest identity, and it holds `SessionSeeAll`/`SessionControlAny`). For `log_after`,
  `open_log`, `open_tree` the ownership check runs *synchronously during the wrapped `.await`* (the
  returned `LogStream`/`TreeStream` is just a receiver; `tree_subscribe` captures the principal at call
  time), so wrapping the call is sufficient.
  - **`tenant_delivery_sse` / `serve_delivery`:** `serve_delivery` (`daemon-delivery/src/lib.rs:99-145`)
    spawns one detached task per owned session that calls `api.subscribe(...)` (122) — wrapping the outer
    call does **not** reach those spawns, and `daemon-delivery` is a pure-contracts crate (no `daemon-host`,
    `Cargo.toml:13-19`) so it cannot set the task-local. **Fix:** in `tenant_delivery_sse`, replace the
    `serve_delivery(...)` call with a small inline per-session loop (daemon-http already depends on
    daemon-host) that spawns each `subscribe` task wrapped in `with_request_context(RequestContext::system(), ...)`.
    `daemon-delivery` stays pure and untouched; the ~15 duplicated lines are flagged for Phase-3
    (capability tokens) to unify. (Alternative surfaced below in Risks.)

### A.4 `roster.rs` — flip `None` from allow to deny + explicit in-process marker

File: `crates/substrate/daemon-host/src/node_api/roster.rs`

- `owner_visible` (347-356): `None => true` → `None => false`.
- `require_session_access` (156-182): the `let Some(principal) = ... else { return Ok(()) }` (163) becomes
  `else { return Err(ApiError::Unauthenticated("no authenticated principal bound to this request".into())) }`.
- Update both doc comments (they currently assert `None` is trusted-in-process) and the unit test
  `no_principal_is_trusted_in_process_and_sees_all` (543-549) → assert `owner_visible(&None, _) == false`;
  add a test that the internal principal (below) sees all.

**Explicit in-process marker (a type, not a bool)** — add to `request_context.rs`:

```rust
pub const INTERNAL_USERNAME: &str = "internal";

impl RequestContext {
    /// The in-process embedded-caller marker: trusted node internals (delivery pumps, ingest,
    /// background input injection) that legitimately cross session ownership. Constructed ONLY
    /// here — never derivable from wire input. Distinct from `system()` (Admin): `internal` holds
    /// the operator-tier session overrides (`SessionSeeAll` + `SessionControlAny`) without `AccessAdmin`.
    pub fn internal() -> Self {
        Self {
            principal: Principal::from_roles("internal", INTERNAL_USERNAME, vec![Role::Operator]),
            origin: None, conn_id: None,
            auth_method: Some(AuthMethod::LocalTrust),
        }
    }
}
```

`Role::Operator` grants exactly `SessionSeeAll` + `SessionControlAny` (`capability.rs:146-154`) — the two
caps `require_session_access`/`owner_visible` consult — plus other operator caps that are irrelevant because
internal callers bypass `authorize` (Auth-2). Distinctness = reserved username `internal`, so audit/logs can
tell it apart from a real operator and from `system`.

**Reserved-username enforcement (daemon-auth, per coordinator addition #2).** In `capability.rs`:
`pub const SYSTEM_USERNAME: &str = "system";`, `pub const INTERNAL_USERNAME: &str = "internal";`,
`pub const RESERVED_USERNAMES: [&str; 2] = [SYSTEM_USERNAME, INTERNAL_USERNAME];`, and a helper
`is_reserved_username(&str) -> bool` (ASCII-case-insensitive). `request_context.rs` re-exports /
references `daemon_auth::{SYSTEM_USERNAME, INTERNAL_USERNAME}` (removing its local `SYSTEM_USERNAME`
definition, keeping the `pub use` in daemon-host `lib.rs` pointed at the daemon-auth constant so the
public name is unchanged). `AuthStore::create_user` returns a new `Error::ReservedUsername` if the
requested username is reserved (and defensively if a minted id equals a reserved name — impossible for a
64-hex id, but cheap). Unit test in `store.rs`: `create_user("system"|"internal", …)` → `Err(ReservedUsername)`;
`create_user("alice", …)` → `Ok`. This is why the `internal` ownership stamp can never be forged by a
real user: no store row can carry that username or id.

### The None-principal call-path inventory (the heart)

Every entry point that reaches an ownership-gated `NodeApiImpl` method (`require_session_access` /
`owner_visible`) **without** a `with_request_context` scope, and its disposition after the flip:

| # | Call path | Location | Class | After the flip |
|---|-----------|----------|-------|----------------|
| 1 | mux `Open(Subscribe)` → `pump_session_log` → `subscribe`/`log_epoch` | `socket.rs:660-671,754-764` | **BUG** (cross-owner transcript) | **Fixed A.2**: pump binds the connection principal → non-owner gets `End{Forbidden}` |
| 2 | mux `Open(EventsSince)` → `pump_node_events` → `events_subscribe` | `socket.rs:734,815` | not session-owned (node feed); harmless | wrapped in connection principal for consistency (A.2) |
| 3 | HTTP `GET …/log` → `log_after` | `daemon-http:258-264` | legit (local-trust) | **Fixed A.3**: wrap `system()` |
| 4 | HTTP SSE/WS subscribe → `open_log` → `subscribe` | `daemon-http:272-324` | legit (local-trust) | **Fixed A.3**: wrap `system()` |
| 5 | HTTP `GET /tree/subscribe` → `tree_subscribe` | `daemon-http:342-364` | legit (local-trust) | **Fixed A.3**: wrap `system()` (else empty tree) |
| 6 | HTTP `GET …/delivery` → `serve_delivery` → `subscribe` (spawned) | `daemon-http:226-233`, `daemon-delivery:99-122` | legit (local-trust) | **Fixed A.3**: inline loop wrapping each spawn in `system()`; `daemon-delivery` untouched |
| 7 | Matrix outbound `DeliveryManager::ensure` (spawned) → `subscribe` + `delivery_targets` | `daemon-matrix/outbound.rs:166-196` | legit internal delivery | **Fix**: wrap spawned task body in `RequestContext::internal()` (matrix depends on daemon-host) |
| 8 | Matrix inbound `on_room_message` → `ingestor.receive` → `submit_routed` | `daemon-matrix/inbound.rs:105`, `daemon-ingest/src/lib.rs:175-187` | legit internal ingest | **Fix**: wrap the `receive(...).await` call in `RequestContext::internal()` (no spawn inside `receive`, so the scope propagates through the awaited `submit_routed`) |
| 9 | Rooms `ensure_subscribed` (spawned) → `subscribe` | `daemon-rooms/adapter.rs:177-185` | legit internal | **Fix**: wrap spawned task body in `RequestContext::internal()` (rooms depends on daemon-host) |
| 10 | Rooms inbound → `ingestor` (`note_turn_*`, reinject) | `daemon-rooms/adapter.rs` | `note_turn_*` are local (no api call); any `submit`/`receive` path | **Fix**: wrap the same as #8/#9 where it calls gated api |
| 11 | Background input injection `inject_session_input` → `self.submit` (live sessions) | `node_api.rs:471-508` (500); callers `fleet/notice_worker.rs:46`, `assembly/mod.rs:305` | legit internal seam | **Fix**: wrap the `self.submit(...)` in `RequestContext::internal()` inside `inject_session_input` (single chokepoint fixes both callers) |
| 12 | Delegated-child owner inheritance | `fleet/job_worker.rs:194-201` | legit; writes `meta.owner` **directly in the store** | **Unaffected** — never calls `require_session_access` |
| 13 | Cron job create owner stamp | `cron.rs:207-209` | writes `owner` field only | **Unaffected** — no gated call; cron worker session dispatch (if any) routes through #11's seam |
| 14 | Builtin command handlers (e.g. `self.cancel`) | `builtins.rs:113` | reached via `command_invoke` dispatch, **which sets context** | **Unaffected** |
| 15 | All wire dispatch (legacy `Call`, mux `Call`, HTTP `POST /api`) | `socket.rs:268-272,633`, `daemon-http:151` | already `with_request_context` | **Unaffected** |

**Blast-radius note:** items 7–11 live outside the literally-listed file scope (daemon-matrix,
daemon-rooms, and `node_api.rs::inject_session_input`). They are **required** — without them the flip
silently breaks reply delivery, inbound ingest, and background injection for live sessions. Matrix/rooms
unit tests use mock `NodeApi` impls (no `require_session_access`), so `cargo test --workspace` may stay
green even if 7–10 are skipped, but production correctness would regress; item 11 may be exercised by
`tests/daemon-conformance` e2e paths and could fail the gate. **Requesting coordinator confirmation to
touch daemon-matrix, daemon-rooms, and `node_api.rs::inject_session_input` as part of this track** (they
share no lines with the ingress-bounds track T2).

**Ownership-stamping side effect (surfaced):** with items 8/11 running under `internal`, sessions created
by chat ingest / injection get `owner = "internal"` (via `note_activity`/`assign` stamping
`current_principal()`), where today they get `owner = None`. Both are hidden from a non-operator network
`User` and visible to `system`/operators, so **roster visibility is unchanged**; the difference is only
that such sessions are now `Owned("internal")` instead of `LegacyUnowned`. Delivery under `internal`
(user_id `"internal"`) then passes via *ownership* for these sessions, not just the `SessionSeeAll` override.

---

## Cluster F — bound ingress before allocation

### F.1 Shared `MAX_FRAME_BYTES`

Add `pub const MAX_FRAME_BYTES: usize = 640 * 1024 * 1024;` in a **new** module
`crates/contracts/daemon-common/src/limits.rs` (daemon-common is the only crate shared by both
`daemon-host`/socket.rs and `daemon-transport`/remote.rs; `remote.rs` already `use daemon_common::...`).
`lib.rs` gets only `pub mod limits;` + `pub use limits::MAX_FRAME_BYTES;` (minimal hunk — the
`child-env-policy` track is concurrently adding its own module to daemon-common, so keep the merge
surface trivial).

**Value rationale (640 MiB):** the largest *legitimate* single frame is a `BlobPut`/`FsWriteFromBlob`
payload, bounded server-side by `MAX_BLOB_SIZE = 256 MiB` (`daemon-host/src/blob_store.rs:18-19`). Rust
`Vec<u8>` serialises as a **CBOR array of ints** (per AGENTS.md — not `bstr`), so each byte ≥ 0x18 costs 2
bytes → a 256 MiB blob is up to ~512 MiB on the wire, plus the `WireC2S::Call`/`ApiRequest::BlobPut`
envelope. 640 MiB (2×256 MiB + 128 MiB headroom) accepts every in-spec frame while cutting the pre-decode
ceiling from the u32 max (**4 GiB → 640 MiB, ~6.4×**). This is a coarse Phase-1 bound; the Phase-4 ingress
governor is where the tighter **pre-auth** cap and **per-transport** caps belong (remote.rs frames are all
tiny control messages and could take a far smaller cap — flagged below).

### F.2 `socket.rs::read_frame` (1101-1114)

```rust
let len = u32::from_be_bytes(len_buf) as usize;
if len > daemon_common::MAX_FRAME_BYTES {
    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "frame too large"));
}
let mut buf = vec![0u8; len];              // now bounded
```

Rejects **before** the `vec![0u8; len]` (1111). No behavior change for in-spec frames.

### F.3 `daemon-transport/src/remote.rs::read_frame` (121-138)

Same guard before `let mut buf = vec![0u8; n];` (133), using `daemon_common::MAX_FRAME_BYTES`.

### F.4 `ws.rs` — WebSocket max message size

File: `crates/substrate/daemon-host/src/ws.rs`

- `accept_mux_upgrade` (118-127) uses `accept_hdr_async` with default tungstenite config. Switch to
  `accept_hdr_async_with_config(stream, callback, Some(cfg))` where
  `cfg = WebSocketConfig { max_message_size: Some(MAX_FRAME_BYTES), max_frame_size: Some(MAX_FRAME_BYTES), ..default }`.
  Factor the config into a small `ws_config()` helper so it is unit-testable and reused by
  `serve_mux_over_ws` (99-113, the already-upgraded single-origin `crate::web` path — confirm the web front
  builds the stream with the same config).
- Defense-in-depth in `WsFrameReader::poll_read` (261-305): the existing guard only rejects
  `payload.len() > u32::MAX` (281-286); tighten it to `> MAX_FRAME_BYTES` so an oversize binary message is
  rejected even if a caller forgets the accept-time config.

---

## Tests (Phase 2 — added FIRST, must fail before the fix, pass after)

Harness: `tests/daemon-conformance/src/node/` (real `NodeApiImpl` via `assemble()`, `serve_api_unix[_authenticated]`,
`MuxApiClient`, `authenticate_scram`) — model on `auth_transport.rs`.

1. **Cross-owner one-shot read (`log_after`) is denied** — new conformance test. Store with `alice`/`bob`
   (`Role::User`). alice authenticates, `Submit(StartTurn)` to session `s` (stamps owner=alice). bob
   authenticates, `Call(Subscribe{s})` → expect `ApiResponse::Error(Forbidden)`. alice `Call(Subscribe{s})`
   → not error. (Repro: today bob reads alice's page.)
2. **Cross-owner mux stream read (`pump_session_log`) is denied** — same setup, bob `open(Subscribe{s})`
   then `next()` → expect `WireS2C::End { error: Some(Forbidden) }` (not an `Item` page). alice
   `open(Subscribe{s})` → gets keepalive/entries. (Repro: today the pump streams alice's log to bob.)
3. **`delivery_targets` cross-owner returns empty** — bob `Call(DeliveryTargets{s})` → empty; alice → real.
4. **HTTP log route still works under local trust after the flip** — build `router(api, true)`, a session
   with entries, `GET /sessions/{s}/log` → `LogPage` (not 401/empty) — proves the `system()` wrap keeps the
   route functional (HTTP has no per-user identity to deny against).
5. **`owner_visible`/internal unit tests** (`roster.rs`): `owner_visible(&None, _) == false`; internal
   principal sees all; existing peer/operator tests unchanged.
6. **Oversize frame rejected before allocation** — unit tests in `socket.rs` and `remote.rs`: feed a reader
   holding only the 4-byte length prefix encoding `MAX_FRAME_BYTES + 1` (and no body); assert
   `read_frame` returns `Err(InvalidData)` immediately (it must error *without* attempting to read the
   oversized body — proven by supplying no body bytes and getting `InvalidData`, not `UnexpectedEof`).
   Also an in-bounds frame round-trips.
7. **WS config caps message size** — `ws.rs` unit test: `ws_config().max_message_size == Some(MAX_FRAME_BYTES)`
   and `max_frame_size == Some(MAX_FRAME_BYTES)`.

---

## Wire / CDDL impact

**None.** No `ApiRequest`/`ApiResponse` variant, field, or any wire-reachable type changes. The internal
principal, the frame cap, and the ownership wraps are all server-side. So `daemon-api.cddl` and
`cargo test -p daemon-api --features arbitrary` are unaffected (will still run in the gate as a no-op check).

---

## Risks / ambiguities (for coordinator)

1. **Scope expansion for the blast radius (items 7–11).** Fixing the flip correctly requires editing
   `daemon-matrix`, `daemon-rooms`, and `node_api.rs::inject_session_input` — outside the literally-listed
   file set but disjoint from track T2 (ingress). Confirm this is in-scope for this track (recommended), or
   the flip must be split from its blast-radius fixes.
2. **`daemon-delivery` purity vs. `tenant_delivery_sse`.** Recommended fix inlines the delivery loop in
   daemon-http (keeps daemon-delivery pure-contracts). Alternative: relocate the request-context primitive
   into a lower crate so `serve_delivery` can carry an internal scope into its spawns — bigger, better
   suited to Phase 3 (capability tokens). Chosen: inline in daemon-http; flag for Phase-3 unification.
3. **`MAX_FRAME_BYTES = 640 MiB` is coarse.** It must exceed 2×`MAX_BLOB_SIZE` to avoid rejecting legit
   max-size `BlobPut` frames, which keeps the *pre-auth* allocation ceiling high (an authenticated-or-local
   surface, but `read_frame` runs before the auth-state check in the mux loop). The tighter pre-auth /
   per-transport split is explicitly the Phase-4 ingress governor. If the coordinator prefers a smaller
   cap now, blob transfer would need chunking (out of scope) or `MAX_BLOB_SIZE` lowering. `remote.rs`
   carries only tiny control frames and would ideally take a much smaller cap — recommend a separate
   `remote`-specific const in Phase 4.
4. **`log_after`/`subscribe` use `control=true` (SessionControlAny), not `control=false` (SessionSeeAll).**
   This mirrors the existing `subscribe` and keeps the `Call`/`Open` forms identical, and is *stricter*
   than a pure read; both operator caps travel together in `Role::Operator`, so no built-in role regresses.
   Noted in case a future custom `SessionSeeAll`-only grant should be able to read live logs.
5. **Ownership stamping** now yields `Owned("internal")` for chat/injection-created sessions instead of
   `LegacyUnowned` (owner-NULL). Net visibility unchanged (both operator-only to network users); surfaced
   for awareness.

---

## Gate (Phase 2, from worktree root)

- `nix develop --command cargo fmt`
- `nix develop --command cargo clippy --workspace --all-targets -- -D warnings`
- `nix develop --command cargo test --workspace`
- `nix develop --command cargo test -p daemon-api --features arbitrary` (no wire change, still run)

Commit on `hardening/auth4-ownership`. Do not merge; do not remove the worktree.
