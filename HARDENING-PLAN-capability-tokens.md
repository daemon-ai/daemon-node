# HARDENING-PLAN — Phase 3 / Cluster A: capability tokens (`AuthorizedFor<Resource>`)

Track: `hardening/capability-tokens` (off `hardening/integration`).
Status: **PLAN ONLY — no source touched yet. Awaiting review/approval before implementing.**

## 0. Goal restated

Phase 1 (`auth4-ownership`, already merged into `hardening/integration`) made the per-resource
ownership check *runtime* and *fail-closed*: every session-touching handler must remember to call
`self.require_session_access(&session, control)` (roster.rs), and a `None` principal now denies.

Cluster A upgrades that from **"every handler must remember to call it"** to **"the type system
won't let you skip it."** The runtime check stays the place the proof is minted; the win is that the
proof (`AuthorizedFor<Session>`) is *un-forgeable* and *required by the signatures* of the
primitives that actually read/mutate a specific session's live state. A new handler that reaches for
session state without first passing the ownership check will **not compile**.

## 1. Architecture as found (why the enforcement lands *below* the trait)

The abstract surface lives in `daemon-api`:

- `SessionApi` / `ControlApi` (composed as `NodeApi`), the object-safe `#[async_trait]` traits every
  transport binds to.
- `dispatch` decodes an `ApiRequest`, calls the trait, encodes the `ApiResponse`. The in-process
  transport calls the trait directly; the socket mux and the C FFI run the same `dispatch`.

Concrete impl: `NodeApiImpl` in `daemon-host` (`crates/substrate/daemon-host/src/node_api/*`).
Ownership is enforced *inside* the trait-impl bodies:

```rust
// session.rs, SessionApi::submit for NodeApiImpl
self.claim(&session, Lifecycle::Live)?;
self.require_session_access(&session, true).await?;   // runtime ownership gate (Phase 1)
self.note_activity(&session, &command).await;
self.live.submit(session, command).await              // <-- touches session state, UNGUARDED by type
```

The state itself lives in `LiveSessions` (`node_api/internals.rs`) — the §17 actor manager. Its
per-session primitives (`submit`, `poll`, `subscribe`, …) are `pub(crate)` and are the physical
"touch session state" surface.

### Key findings that shape the design

1. **Every external caller goes *through the trait*, not through `LiveSessions`.**
   - The socket mux stream pump (`socket.rs::pump_session_log`) calls `api.log_epoch(...)` /
     `api.subscribe(...)` on `Arc<dyn NodeApi>`, inside a re-established `with_request_context(ctx, …)`
     scope so `require_session_access` sees the connection's principal.
   - The HTTP adapter (`daemon-http/src/lib.rs`) likewise calls `api.log_after` / `api.subscribe`
     under a `RequestContext::system()` scope.
   - The FFI, conformance harness, matrix/ingest/delivery adapter tests, and `daemon-api`'s own
     `Bare` demo all bind to the **trait**.
   Confirmed: `LiveSessions` is `pub(crate)` and `NodeApiImpl.live` is a module-private field, so the
   guarded primitives are reachable **only from within the `node_api` module tree** — never from
   another crate, never from another daemon-host module.

2. **Touching the trait signatures is the wrong move.** There are 7+ `impl SessionApi`/`impl ControlApi`
   blocks across crates (`daemon-core-ffi`, `daemon-api::Bare`, and the matrix/ingest/http/delivery
   test mocks). Threading a witness param into the trait would (a) break all of them, (b) require the
   witness to be `pub` and constructible by `dispatch` in `daemon-api` — which is exactly where it
   would become *forgeable* — and (c) risk object-safety. **So the trait is left byte-for-byte
   unchanged.** Enforcement lands at the `NodeApiImpl` (trait body) → `LiveSessions` (state)
   boundary, wholly inside `daemon-host`.

3. **`#![forbid(unsafe_code)]` is set in `daemon-host/src/lib.rs`.** A witness struct with a private
   field therefore cannot be forged by `transmute`/`mem::zeroed` anywhere in the crate — the
   un-forgeability argument is airtight in safe Rust.

## 2. Witness type design (`AuthorizedFor<Resource>`)

New module `crates/substrate/daemon-host/src/node_api/authorized.rs`, wired as `mod authorized;` in
`node_api.rs`. It owns the witness type **and** the minting function so the constructor stays
module-private.

```rust
// authorized.rs
use daemon_common::SessionId;
use std::marker::PhantomData;

/// Resource-class marker for a session capability token. An empty enum (uninhabited, zero-sized,
/// never constructed) used only as the `Resource` type parameter of `AuthorizedFor`. Kept in this
/// module so `AuthorizedFor<Session>` reads naturally; renamed to `SessionResource` if the bare
/// `Session` name is judged ambiguous next to `SessionId`/`SessionApi` (no collision exists today —
/// no bare `Session` type is imported in `node_api`).
pub(crate) enum Session {}

/// A capability token proving the Auth-4 per-resource ownership check passed for one specific
/// resource of class `R`. Carries the `SessionId` it authorizes so a guarded primitive derives the
/// target from the *proof*, never from a separately-passed id that could disagree with what was
/// checked (no authorize-A-act-on-B).
///
/// **Un-forgeable.** The single field is private and the only constructor (`mint`) is private to
/// this module; the sole `pub(crate)` producer is the ownership check `require_session_access`
/// below (and the read-minting `authorize_read`). Anywhere else in `daemon-host` you may *name*,
/// *hold*, and *pass* an `AuthorizedFor<Session>`, but you cannot *create* one without passing the
/// check. `#![forbid(unsafe_code)]` (crate-level) rules out a transmute backdoor.
pub(crate) struct AuthorizedFor<R> {
    session: SessionId,
    _resource: PhantomData<R>,
}

impl<R> AuthorizedFor<R> {
    /// The session this token authorizes (guarded primitives read the id from here).
    pub(crate) fn session(&self) -> &SessionId {
        &self.session
    }
}

impl AuthorizedFor<Session> {
    /// Module-private mint. NOT `pub(crate)`: only the ownership checks in this module can call it,
    /// so a token can only come into existence *after* a passing check.
    fn mint(session: SessionId) -> Self {
        Self { session, _resource: PhantomData }
    }
}
```

Design notes / decisions:

- **Generic `AuthorizedFor<R>` with a `Session` marker** matches the plan's `AuthorizedFor<Resource>`
  and future-proofs for `AuthorizedFor<Profile>` / `<Cron>` later, at zero cost today.
- **Carries the `SessionId`** (design "A"): strictly stronger than a zero-sized token — the guarded
  primitive uses `auth.session()`, so a caller cannot gate session A and then act on session B. This
  removes the redundant `session:` argument from the guarded primitives (see §3).
- **No `#[must_use]`.** The enforcement is that the downstream method *requires* the token as a
  parameter; `#[must_use]` would only add noise at the gate-only sites that legitimately discard the
  token (`session_history`, and any handler that gates but calls no guarded primitive). Not needed.
- **Not `Clone`/`Copy`/`Default`/`Serialize`** — nothing needs them, and omitting `Default` removes
  the one "construct from nothing" escape. (Cloning a *valid* token would be harmless, but there's no
  call for it; keep the surface minimal.)
- The witness is **`pub(crate)`, never `pub`** — it is an internal enforcement mechanism with no
  meaning outside `daemon-host` (it's un-constructable externally and no external caller reaches the
  guarded methods). Exporting it would be misleading.

### Minting: the ownership check returns the token

`require_session_access` (moved from `roster.rs` into `authorized.rs` so it can call the private
`mint`) changes its return type from `Result<(), ApiError>` to
`Result<AuthorizedFor<Session>, ApiError>`. Its logic is factored so the *decision* is a pure,
store-free function (unit-testable without a full `NodeApiImpl`):

```rust
// authorized.rs
use super::roster::SessionOwnership;

/// Pure ownership decision → token. The single mint site for the interaction/read gate. Store-free
/// (takes the already-resolved `SessionOwnership`), so it is directly unit-testable.
fn authorize_ownership(
    session: &SessionId,
    principal: &Option<daemon_auth::Principal>,
    control: bool,
    ownership: SessionOwnership,
) -> Result<AuthorizedFor<Session>, ApiError> {
    let Some(principal) = principal else {
        return Err(ApiError::Unauthenticated(
            "no authenticated principal bound to this request".into(),
        ));
    };
    let override_cap = if control {
        daemon_auth::Capability::SessionControlAny
    } else {
        daemon_auth::Capability::SessionSeeAll
    };
    if principal.has(override_cap) {
        return Ok(AuthorizedFor::mint(session.clone()));
    }
    match ownership {
        SessionOwnership::Absent => Ok(AuthorizedFor::mint(session.clone())),
        SessionOwnership::Owned(owner) if owner == principal.user_id => {
            Ok(AuthorizedFor::mint(session.clone()))
        }
        _ => Err(ApiError::Forbidden(format!(
            "session {session} is not owned by the caller"
        ))),
    }
}

impl NodeApiImpl {
    pub(crate) async fn require_session_access(
        &self,
        session: &SessionId,
        control: bool,
    ) -> Result<AuthorizedFor<Session>, ApiError> {
        let principal = crate::request_context::current_principal();
        // Short-circuit the store read for the override-cap / no-principal cases, exactly as today.
        let ownership = self.session_ownership(session).await;
        authorize_ownership(session, &principal, control, ownership)
    }
}
```

`session_ownership` and `SessionOwnership` stay in `roster.rs` (they are `pub(crate)`, reachable
from the new module). `require_operator` and the `owner_visible` free function stay in `roster.rs`
unchanged — they do **not** mint (they gate operator-tier mutations and enumeration filters
respectively, neither of which reaches a guarded per-session primitive except `session_get`; see
§3).

Behavior parity note: today `require_session_access` returns early *before* the `session_ownership`
store read when the override cap is present or the principal is absent. To preserve that (avoid a
needless store read on the operator/None path), the real implementation will keep the early
`current_principal()`/override branch inline and only call `session_ownership` on the
owner-comparison path — i.e. `authorize_ownership` will be split so the store read stays lazy. The
pure-decision variant above is the shape the unit tests exercise (they pass a pre-resolved
`SessionOwnership`); the method wrapper keeps the lazy read. Net: **no behavior change** vs Phase 1,
only the return type gains the token.

## 3. Which signatures change (exact blast radius)

All within `daemon-host/src/node_api/`. No wire type, no `daemon-api` trait, no other crate.

### 3a. Guarded `LiveSessions` primitives (`internals.rs`) — add `auth: &AuthorizedFor<Session>`, derive the id from it

For the two that take `session: SessionId` by value today (`submit`, `submit_from`), drop that
parameter and bind `let session = auth.session();` at the top so the body is otherwise unchanged
(the body only ever uses `session` by reference). For the rest that take `session: &SessionId`, drop
that parameter and bind the same local.

| primitive | control? | today | after |
|---|---|---|---|
| `submit` | write | `(session: SessionId, command)` | `(auth: &AuthorizedFor<Session>, command)` |
| `submit_from` | write | `(session: SessionId, origin, command)` | `(auth: &AuthorizedFor<Session>, origin, command)` |
| `poll` | write | `(session: &SessionId, max)` | `(auth: &AuthorizedFor<Session>, max)` |
| `respond` | write | `(session: &SessionId, response)` | `(auth: &AuthorizedFor<Session>, response)` |
| `log_after` | write | `(session: &SessionId, after_seq, max)` | `(auth: &AuthorizedFor<Session>, after_seq, max)` |
| `subscribe` | write | `(session: &SessionId, after_seq)` | `(auth: &AuthorizedFor<Session>, after_seq)` |
| `log_epoch` | read | `(session: &SessionId)` | `(auth: &AuthorizedFor<Session>)` |
| `delivery_targets` | read | `(session: &SessionId)` | `(auth: &AuthorizedFor<Session>)` |
| `handover` | write | `(session: &SessionId, target)` | `(auth: &AuthorizedFor<Session>, target)` |
| `record_meta` | write | `(args: RecordMetaArgs)` | `(auth: &AuthorizedFor<Session>, args)` — use `auth.session()` for the map lookup; keep `args.origin/kind/body` |
| `interrupt` | write | `(session: &SessionId)` | `(auth: &AuthorizedFor<Session>)` |
| `rewind_resident` | write | `(session: &SessionId, anchor, restore_workspace)` | `(auth: &AuthorizedFor<Session>, anchor, restore_workspace)` |

Internal recursion inside `LiveSessions`: `submit` calls `submit_from` — forward `auth`. `submit_from`
calls `ensure`, `seed_primary`, `record_inbound`, `existing`, `sessions.remove` — all **unguarded**
helpers (not per-session ownership ops), so they keep taking `&SessionId` via `auth.session()`.

### 3b. Trait-impl handlers (`session.rs`, `control.rs`) — capture the token, pass `&auth`

Each of these already calls `require_session_access` immediately before touching state; the change is
`let auth = self.require_session_access(&session, …).await?;` then pass `&auth` to the guarded call.

`session.rs` (`impl SessionApi`):
- `submit` (L8), `submit_from` (L19), `submit_as` (L85), `submit_routed` (L106): capture token, pass to
  `self.live.submit`/`submit_from`. `note_activity`, `ensure`, `seed_primary_target` stay unguarded.
- `poll` (L141), `respond` (L147), `handover` (L221), `record_meta` (L227): capture token, pass through.
- `log_after` (L167), `subscribe` (L182): capture token, pass through.
- Non-fallible reads that gate with `if …is_err() { return default }`:
  - `session_history` (L152): gates but calls `self.read_history(...)` (journal store, **not** a
    guarded primitive) → keep the `.is_err()` gate form, token discarded (no `#[must_use]`, no warn).
  - `log_epoch` (L188): `let Ok(auth) = self.require_session_access(&session,false).await else { return 0 };`
    then `self.live.log_epoch(&auth)`.
  - `delivery_targets` (L198): same pattern → `self.live.delivery_targets(&auth)`.

`control.rs` (`impl ControlApi`):
- `cancel` (L369): capture token at L371 → `self.live.interrupt(&auth)`.
- `rewind` (L1328): capture token at L1334 → `self.live.rewind_resident(&auth, …)`.
- `session_get` (L162): **convert** the `owner_visible(&current_principal(), &meta.owner)` gate
  (L171) to `let Ok(auth) = self.require_session_access(&session, false).await else { return None };`
  then `self.live.delivery_targets(&auth)` at L188. Equivalent for the reachable cases (the `Absent`
  session is already returned as `None` earlier at L165-167, and `SessionSeeAll`/owner map identically
  between `owner_visible` and `require_session_access(false)`). Costs one extra `session_meta` read on
  this detail-pane path (acceptable, non-hot); the alternative — a second minter keyed off an already
  known owner — would reintroduce a "construct without check" seam, which defeats the purpose.

### 3c. Deliberately NOT guarded (documented residual coverage)

- **`conv_view`** (`internals.rs::LiveSessions::conv_view`) reads a resident session's transcript. It
  is reached from (i) `session_recap` (control.rs L241) — already gated by `owner_visible`, and (ii)
  `NodeApiImpl::live_conv_view` (a **`pub`** method), which is called from **`bins/daemon/src/main.rs`
  L1584** (the `session_search` agent-tool archive reader). Because that caller lives in another
  crate, it cannot mint a daemon-host-private token, so `conv_view` cannot require the witness without
  either leaking a public constructor (forgeable) or breaking the tool. `conv_view` therefore stays
  outside the witnessed set. Residual posture: `session_recap` keeps its `owner_visible` gate; the
  agent-tool archive path keeps its existing (in-process, tool-scoped) posture. Called out here as a
  known gap for a later phase (a `conv_view` witness would want a `NodeApi` trait read op the tool
  calls through, which is out of Cluster A's scope).
- **Infrastructural `LiveSessions` methods** that are *not* per-session ownership ops stay unchanged:
  `register_delivery_sink`/`unregister_delivery_sink`, `all_primary_targets`/`push_to_target` (cron
  broadcast fan-out across *all* sessions), `delivery_sessions` (transport-owned enumeration),
  `live_ids`, `is_resident`, `resident_is_foreign`, `handle_if_live`, `ensure`, `seed_primary`/
  `seed_primary_target`, `record_inbound`, `existing`, and the `set_*` wiring setters. None reads or
  mutates *one caller-scoped* session's private state on behalf of a request principal; forcing a
  token on them would be miscategorization (there is no single owning principal for a cron broadcast).
- **Enumeration filters** (`roster_scoped`, `tree_owned`, `session_search`, `session_get`'s
  existence probe) keep `owner_visible` — they filter *sets* of rows, minting one token per row would
  be meaningless. Their per-row visibility remains the Phase 1 runtime check.

## 4. Object-safety, wire, and composition

- **Object-safety: unaffected.** The public `NodeApi`/`SessionApi`/`ControlApi` traits are untouched,
  so `Arc<dyn NodeApi>` still works everywhere. `LiveSessions` is a concrete struct — no dyn concern.
- **Wire types: none change.** The witness is a compile-time, in-process token; it is never
  serialized, never named in `ApiRequest`/`ApiResponse`, never in `daemon-api.cddl`. Confirmed by
  scope (this is internal handler plumbing). The `arbitrary`/conformance gate is still run per the
  contract (see §7) to *prove* no wire drift crept in.
- **Composition with the Phase 1 runtime check (no duplication).** The runtime check is not
  duplicated — it is *reused as the mint site*. `require_session_access` still performs exactly the
  same decision it does today; it now returns the proof of that decision instead of `()`. The
  compile-time layer adds *nothing at runtime* (the token is zero-cost beyond a `SessionId` clone it
  already effectively did) and simply makes the existing check un-skippable by construction. `None`
  stays fail-closed (no token minted); `system()`/`internal()` request contexts still mint via the
  override-cap branch, so the mux/HTTP/ingest/delivery pumps keep working unchanged.

## 5. Tests

Added tests-first (before the signature churn), in a `#[cfg(test)] mod` in `authorized.rs`.

### 5a. Runtime mint tests (the token is minted iff the check passes) — the core new coverage

Exercise the pure `authorize_ownership(session, principal, control, ownership)`:

1. `owner_mints_token`: `Owned("alice")` + principal alice + `control=true/false` → `Ok`, and
   `token.session() == session` (proves the token carries the checked id).
2. `non_owner_is_denied_no_token`: `Owned("bob")` + principal alice → `Err(Forbidden)` (no token).
3. `none_principal_is_denied_no_token`: principal `None` → `Err(Unauthenticated)` (fail-closed,
   mirrors Phase 1).
4. `legacy_unowned_denied_for_non_operator`: `LegacyUnowned` + plain user → `Err`.
5. `see_all_override_mints_for_read`: principal with `SessionSeeAll`, `control=false`,
   `Owned("bob")` → `Ok` (operator read override).
6. `control_any_override_mints_for_write`: principal with `SessionControlAny`, `control=true`,
   `Owned("bob")` → `Ok` (operator interaction override).
7. `absent_session_mints`: `Absent` → `Ok` (the create/not-found path still runs downstream).
8. `internal_marker_mints_like_operator`: `RequestContext::internal().principal` crosses ownership
   (the delivery/ingest/injection pumps).

These need no store/`NodeApiImpl` — that is the reason for factoring the pure decision function (the
crate has no existing lightweight `NodeApiImpl` test harness; heavyweight construction is avoided).

### 5b. Compile-fail demonstration (documented — no new dependency)

The user's contract explicitly allows a *documented* compile-fail case. Rationale for not using
`trybuild`: the witnessed methods and the token are `pub(crate)`, so a `trybuild` file (compiled as a
separate crate) and a rustdoc `compile_fail` doctest (also a separate crate, and only generated for
`pub` items) can see *neither* the guarded methods nor the token — a real automated compile-fail for
an internal, pub(crate) mechanism is not expressible without either making the surface `pub` (leaks an
un-constructable type into the public API) or adding `trybuild` (a new dependency → Nix change, which
this dep-free track avoids). Instead:

- A `//` documented compile-fail example in the `authorized.rs` module docs showing the exact code a
  future handler might write and *why it cannot compile*:

  ```text
  // Will NOT compile — there is no way to obtain an `AuthorizedFor<Session>` except from
  // `require_session_access` / `authorize_read`:
  //
  //   self.live.submit(&AuthorizedFor::mint(session), command)  // `mint` is private to this module
  //   self.live.submit(session, command)                        // wrong arg type: expected &AuthorizedFor<Session>
  //
  // The only path that type-checks is:
  //   let auth = self.require_session_access(&session, true).await?;
  //   self.live.submit(&auth, command).await
  ```

- The **structural guarantee is reviewable and enforced by the compiler for real code**: the private
  field + module-private `mint` mean the whole workspace build (the gate's `cargo build`/`clippy`)
  *is* the compile-time proof — if any daemon-host code tried to call a guarded primitive without a
  token, `cargo clippy -D warnings` in the gate would fail. The negative example is documentation of
  a property the compiler already enforces on every build.

- **Optional follow-up (flagged for the coordinator, NOT in this track):** if a hard, standalone
  automated compile-fail gate is wanted, add `trybuild` as a dev-dependency (needs a Nix devShell dep
  addition) and a `tests/compile_fail/*.rs` that calls a guarded method without a token. I recommend
  deferring this to the Phase 4 `clippy-disallow` track (which already owns "turn conventions into
  build breaks" and touches tooling), rather than adding a dep here.

### 5c. Regression / behavior-parity

- Existing Phase 1 ownership tests (`roster.rs::owner_visible_tests`, `authz.rs` role matrix) stay
  green unchanged — the runtime decision is identical.
- The full `--workspace` test run exercises the trait-through paths (mux/http/ingest/delivery mocks,
  ffi) that call the unchanged trait, proving the below-the-trait refactor did not alter observable
  behavior.

## 6. Residual coverage / known limits (explicit)

1. `conv_view` / `live_conv_view` (session_search archive tool, `bins/daemon`) — not witnessed; see
   §3c. Owner scoping there is via `owner_visible` (recap) or the tool's own in-process posture.
2. The witness proves *"an ownership check passed for this session id"*, not *"the principal is X"* —
   it deliberately does not carry the principal (the check already resolved the answer; carrying the
   id is enough to prevent A/B confusion). Cross-owner enumeration filters remain the runtime
   `owner_visible` path.
3. Operator overrides (`SessionSeeAll`/`SessionControlAny`) and the synthetic `system`/`internal`
   principals still mint tokens (by design) — the token means "authorized," which includes the
   legitimate override/embedded paths.
4. Scope is `AuthorizedFor<Session>` only; `Profile`/`Cron`/`Fs` resources are future markers on the
   same generic, not implemented here.

## 7. Exact gate (from worktree root, inside the devShell)

Tests first, then minimal hunks; paste tails of each:

```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo test -p daemon-api --features arbitrary
```

Machine-load caveat (from the track brief): concurrent Opus builds can starve the timeout-sensitive
`bins/daemon/tests/host_launch.rs` under a full parallel `--workspace` run. If those fail, re-run
isolated before treating as real:

```
nix develop --command cargo test -p daemon --test host_launch -- --test-threads=2
```

Known pre-existing flakes to ignore (not caused by this change): `detached_delegation` (×2),
`process_notify` store-seam. Only new/different signatures are real. The `arbitrary` conformance run
is expected to be a **no-op pass** (no wire change) — it is run to *prove* that.

Do **not** merge; do **not** remove the worktree (coordinator does both in wave order).

## 8. Implementation order (when approved)

1. Add `authorized.rs`: `AuthorizedFor<R>` + `Session` marker + private `mint` + `authorize_ownership`
   pure fn + the `#[cfg(test)]` mint tests + the documented compile-fail example. Wire `mod authorized;`
   and re-export `pub(crate) use authorized::{AuthorizedFor, Session};` in `node_api.rs`.
2. Move `require_session_access` from `roster.rs` into `authorized.rs`, changing its return type to
   `Result<AuthorizedFor<Session>, ApiError>` (lazy store read preserved). Update the roster.rs doc
   cross-reference.
3. Change the 12 guarded `LiveSessions` signatures in `internals.rs` (§3a); fix internal recursion.
4. Update the ~15 call sites in `session.rs` / `control.rs` (§3b) to capture and pass `&auth`;
   convert `session_get`'s gate (§3b).
5. Run the gate (§7); paste tails.

Each step compiles independently except 3↔4 (the signature change and its call sites land together);
keep those as one tight commit-shaped hunk per file.
