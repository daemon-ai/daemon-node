# Auth 2 — authz core (Track A) — implementation plan

Worktree: `/home/j/experiments/daemon-worktrees/auth2-authz-core` (branch `feature/auth2-authz-core`).
Phase-1 deliverable: this plan. **No source changed yet.** Awaiting coordinator validation.

Scope (pure backend, no wire/CDDL change):

1. **request-context** — a task-local request context + `with_request_context`/`current_principal`,
   fail-closed (unset = no principal = deny), plus a deliberate local-trust `system()` principal.
2. **authorize-gate** — an exhaustive `required_capability(&ApiRequest) -> Capability` (no `_` arm)
   + `authorize(&ApiRequest) -> Result<(), ApiError>` reading the task-local; invert the
   `commands.rs` `None => Admin` polarity.

---

## 1. Investigation summary (what's already in place)

- **`crates/substrate/daemon-auth`** (`capability.rs`): `Capability` (24 snake_case variants),
  `Role` (`Viewer < User < Operator < Admin`, `ALL`, `capabilities()`, `from_wire`/`as_str`),
  `Principal { user_id, username, roles, capabilities: BTreeSet<Capability> }` with
  `from_roles(..)`, `.has(cap)`, `.can_see_all_sessions()`. Re-exported from the crate root
  (`pub use capability::{Capability, Principal, Role}`).
  - Role grants (authoritative, from `Role::capabilities()`):
    - Viewer = all `*Read` (`SessionRead, ControlRead, FleetRead, ModelsRead, ProfileRead,
      CredentialRead, CronRead, RoutingRead, MessagingRead, RegistryRead, FsRead`).
    - User = Viewer + (`SessionWrite, ProfileWrite, CredentialWrite, CronWrite, MessagingWrite,
      FsWrite`).
    - Operator = User + (`SessionSeeAll, SessionControlAny, ControlWrite, FleetWrite, ModelsWrite,
      RoutingWrite, RegistryWrite`).
    - Admin = Operator + `AccessAdmin`.
- **`crates/contracts/daemon-api/src/wire.rs`**: `ApiError::Unauthenticated(String)` /
  `Forbidden(String)` exist (frozen v2). `ApiRequest` has **148 variants**. `PrincipalView` exists
  (advisory client mirror; not used by the gate).
- **`crates/contracts/daemon-api/src/dispatch.rs`**: 12 `serve_*` helpers fan out over disjoint
  `ApiRequest` subsets; `dispatch` chains them and ends in `unreachable!`. Every variant is routed
  by exactly one helper (conformance-verified). The gate's match mirrors these groups one-to-one.
- **`crates/substrate/daemon-telemetry/src/trace.rs`**: the task-local pattern to mirror —
  `tokio::task_local!`, `with_trace(id, fut).await` (`TRACE.scope(..)`), `current_trace()` via
  `try_with(..).unwrap_or(NONE)`. We mirror the **scope/`try_with`** shape (no interior `Cell`: the
  context is set once per request, never rewritten mid-scope).
- **`crates/substrate/daemon-host/src/commands.rs:211`**: `caller_access(origin)` returns
  `CommandAccess::Admin` for `origin == None` — the inverted polarity. Only caller is
  `node_api/control.rs:1232` (`command_invoke`). Test at `commands.rs:465` asserts the old polarity
  and must be updated.
- **Deps**: `daemon-host/Cargo.toml` already depends on `daemon-api`, `daemon-protocol`,
  `daemon-telemetry` — but **not** `daemon-auth`. The workspace root `Cargo.toml`
  `[workspace.dependencies]` has **no** `daemon-auth` entry; both must be added.

---

## 2. Module location (decision + justification)

**Live in `daemon-host` as two new sibling modules** (server-side, where the task-local boundary and
the dispatch gate run):

- `crates/substrate/daemon-host/src/request_context.rs`
- `crates/substrate/daemon-host/src/authz.rs`

Justification:
- The gate references `ApiRequest`/`ApiError` (`daemon-api`) **and** `Capability`/`Principal`
  (`daemon-auth`). `daemon-host` is the lowest crate that already (or will) depend on both;
  `daemon-api` must stay transport-agnostic and must not gain a `daemon-auth` dep, and `daemon-auth`
  must stay protocol-agnostic (its own docstring says the gate "lives in `daemon-api`/`daemon-host`").
- Track B's `serve_mux` integration also lives in `daemon-host` (`socket.rs`), so it consumes these
  symbols in-crate with no new cross-crate edge.
- A task-local is process-global; co-locating it with the server that establishes/reads it keeps the
  fail-closed boundary auditable in one place.

---

## 3. Files to create / edit

| File | Action |
|---|---|
| `Cargo.toml` (workspace root) | **edit**: add `daemon-auth = { path = "crates/substrate/daemon-auth" }` to `[workspace.dependencies]` (next to the other substrate crates ~L180–186). |
| `crates/substrate/daemon-host/Cargo.toml` | **edit**: add `daemon-auth = { workspace = true }` to `[dependencies]`. |
| `crates/substrate/daemon-host/src/request_context.rs` | **new**: `RequestContext`, `AuthMethod`, task-local, `with_request_context`, `current_principal`, `current_context`. |
| `crates/substrate/daemon-host/src/authz.rs` | **new**: `required_capability`, `authorize`. |
| `crates/substrate/daemon-host/src/lib.rs` | **edit**: add `pub mod authz; pub mod request_context;` and re-exports. |
| `crates/substrate/daemon-host/src/commands.rs` | **edit**: invert `caller_access` (L211–216) to read `current_principal()`; fix the `caller_access(None)` test (L465). |

---

## 4. Exact signatures

### `request_context.rs`

```rust
use crate::request_context_imports::*; // illustrative; see below
use daemon_auth::{Capability, Principal, Role};
use daemon_protocol::Origin;
use std::future::Future;

tokio::task_local! {
    static REQUEST_CONTEXT: RequestContext;
}

/// How the principal bound to this request was authenticated (advisory; audit/telemetry).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthMethod {
    /// In-process / FFI / local Unix socket trust (no network auth performed).
    LocalTrust,
    Scram,
    Plain,
    External, // mTLS client cert
    Token,    // AuthResume / server token
}

/// The identity + provenance bound to one in-flight request. Established once (post-auth, or by a
/// local-trust site) and read by the gate. Absence of a context = no principal = DENY (fail-closed).
#[derive(Clone, Debug)]
pub struct RequestContext {
    pub principal: Principal,
    pub origin: Option<Origin>,
    pub conn_id: Option<u64>,
    pub auth_method: Option<AuthMethod>,
}

impl RequestContext {
    /// A network-authenticated principal (Track B's primary entry point).
    pub fn authenticated(principal: Principal, origin: Option<Origin>) -> Self;

    /// The deliberate local-trust principal: full `Role::Admin` capability set. Construct ONLY at
    /// in-process / FFI / local-Unix sites that are trusted by deployment (never on a TCP path).
    pub fn system() -> Self; // principal = Principal::from_roles("system","system",[Role::Admin]);
                             // origin = None; auth_method = Some(LocalTrust)

    pub fn with_conn_id(self, conn_id: u64) -> Self;
    pub fn with_auth_method(self, auth_method: AuthMethod) -> Self;
}

/// Run `fut` with `ctx` bound as the task-local request context.
pub async fn with_request_context<F>(ctx: RequestContext, fut: F) -> F::Output
where
    F: Future;

/// The principal bound to the current request, or `None` when no context is active (fail-closed).
pub fn current_principal() -> Option<Principal>;   // REQUEST_CONTEXT.try_with(|c| c.principal.clone()).ok()

/// The full current context (origin/conn_id/auth_method for ownership + audit), or `None`.
pub fn current_context() -> Option<RequestContext>; // REQUEST_CONTEXT.try_with(Clone::clone).ok()
```

### `authz.rs`

```rust
use crate::request_context::current_principal;
use daemon_api::{ApiError, ApiRequest};
use daemon_auth::Capability;

/// The single capability gating each request. ONE exhaustive match, NO `_` arm — adding an
/// `ApiRequest` variant without a mapping is a compile error (the build-time exhaustiveness guard).
pub fn required_capability(req: &ApiRequest) -> Capability;

/// Capability gate over the task-local principal. `None` -> `Unauthenticated`; present-but-missing
/// the capability -> `Forbidden`. (Per-resource ownership is enforced later by Track C.)
pub fn authorize(req: &ApiRequest) -> Result<(), ApiError>;
//  let need = required_capability(req);
//  match current_principal() {
//      None                      => Err(ApiError::Unauthenticated("no authenticated principal".into())),
//      Some(p) if p.has(need)    => Ok(()),
//      Some(_)                   => Err(ApiError::Forbidden(format!("missing capability: {need:?}"))),
//  }
```

### `lib.rs` additions

```rust
pub mod authz;
pub mod request_context;
pub use authz::{authorize, required_capability};
pub use request_context::{current_context, current_principal, with_request_context, AuthMethod, RequestContext};
```

### `commands.rs` inversion (replaces L211–216)

```rust
/// The caller's command-access tier, derived from the **authenticated principal** (not from origin
/// presence). Admin tier requires an operator/admin principal (`ControlWrite` or `AccessAdmin`);
/// everyone else — including an unauthenticated/empty context — gets the read-only `User` floor.
/// Fail-closed: the absence of identity never yields the admin tier (was `None => Admin`).
pub fn caller_access(_origin: Option<&Origin>) -> CommandAccess {
    match crate::request_context::current_principal() {
        Some(p)
            if p.has(daemon_auth::Capability::AccessAdmin)
                || p.has(daemon_auth::Capability::ControlWrite) =>
        {
            CommandAccess::Admin
        }
        _ => CommandAccess::User,
    }
}
```

The local CLI/FFI/Unix paths keep admin because they run inside `with_request_context(system())`
(its `Role::Admin` principal has both caps). `command_invoke` (`node_api/control.rs:1232`) is
unchanged: it still calls `caller_access` + `access_allows`, now principal-driven.

---

## 5. Full `ApiRequest` → `Capability` mapping (all 148 variants, grouped by `serve_*`)

Mapping rule: the coarse capability for the *kind* of action (read vs write) on the surface the
request belongs to, taking the `Capability` docstrings as authoritative. Per-resource ownership
(own vs. any session) is **not** decided here — that's the `SessionSeeAll`/`SessionControlAny`
override enforced by Track C; no variant maps to those override caps.

**session** (`serve_session`): `Submit`→SessionWrite · `SubmitRouted`→SessionWrite · `Poll`→SessionRead · `Respond`→SessionWrite · `SessionHistory`→SessionRead · `Subscribe`→SessionRead · `DeliveryTargets`→SessionRead · `DeliverySessions`→SessionRead · `Handover`→SessionWrite · `RecordMeta`→SessionWrite · `SetSessionModel`→SessionWrite · `SetSessionMode`→SessionWrite · `SetSessionOverlay`→SessionWrite

**control** (`serve_control`): `Health`→ControlRead · `Stats`→ControlRead · `Telemetry`→ControlRead · `Sessions`→SessionRead · `ApprovalsPending`→ControlRead · `ApprovalDecide`→ControlWrite · `CheckpointList`→ControlRead · `EventsSince`→ControlRead · `CheckpointRewind`→ControlWrite · `Assign`→ControlWrite · `Cancel`→ControlWrite · `VerifyingKey`→ControlRead · `SessionsQuery`→SessionRead · `SessionGet`→SessionRead · `SessionsByProfile`→SessionRead · `SessionSearch`→SessionRead · `SessionUpdateMeta`→SessionWrite *(reviewable, see fork #5)* · `Rewind`→ControlWrite

(`ControlRead`/`ControlWrite` docstrings explicitly enumerate health/stats/telemetry/approvals/checkpoints and assign/cancel/approvals/rewind respectively — followed verbatim. Roster reads are `SessionRead` because the roster is "one's own sessions".)

**fleet** (`serve_fleet`): `Fleet`→FleetRead · `Tree`→FleetRead · `Unit`→FleetRead · `UnitEvents`→FleetRead · `UnitOutbound`→FleetRead · `UnitHistory`→FleetRead · `Pause`→FleetWrite · `Resume`→FleetWrite · `Scale`→FleetWrite

**models** (`serve_models`): `ModelSearch`→ModelsRead · `ModelFiles`→ModelsRead · `ModelDownload`→ModelsWrite · `ModelDownloads`→ModelsRead · `ModelCancel`→ModelsWrite · `ModelPause`→ModelsWrite · `ModelResume`→ModelsWrite · `ModelCatalog`→ModelsRead · `ModelDelete`→ModelsWrite · `ModelActivate`→ModelsWrite · `ModelRecommend`→ModelsRead · `ModelQuantize`→ModelsWrite · `ModelQuantizes`→ModelsRead · `ModelInspect`→ModelsRead · `Models`→ModelsRead · `ModelCurrent`→ModelsRead

**profile** (`serve_profile`): `ProfileList`→ProfileRead · `ProfileGet`→ProfileRead · `ProfileCreate`→ProfileWrite · `ProfileUpdate`→ProfileWrite · `ProfileDelete`→ProfileWrite · `ProfileSelect`→ProfileWrite · `ProfileClone`→ProfileWrite · `ProfileExport`→ProfileRead · `ProfileImport`→ProfileWrite · `ProfileHistory`→ProfileRead · `ProfileAt`→ProfileRead · `ProfileRevert`→ProfileWrite · `SkillHistory`→ProfileRead · `SkillAt`→ProfileRead · `SkillRevert`→ProfileWrite · `SkillGet`→ProfileRead · `SkillPut`→ProfileWrite

**curator** (`serve_curator`): `CuratorList`→ProfileRead · `CuratorPin`→ProfileWrite · `CuratorUnpin`→ProfileWrite · `CuratorArchive`→ProfileWrite · `CuratorRestore`→ProfileWrite · `CuratorRun`→ProfileWrite

**auth/credential** (`serve_auth`): `AuthBegin`→CredentialWrite · `AuthComplete`→CredentialWrite · `AuthCancel`→CredentialWrite · `AuthProviders`→CredentialRead · `CredentialSet`→CredentialWrite · `CredentialList`→CredentialRead · `CredentialRemove`→CredentialWrite

(`CredentialWrite` docstring: "…and run interactive (OAuth) auth flows" — the `Auth*` flow ops map there. These are provider/OAuth credential flows, distinct from the SASL login handshake Track B owns at the transport layer.)

**cron** (`serve_cron`): `CronList`→CronRead · `CronCreate`→CronWrite · `CronUpdate`→CronWrite · `CronDelete`→CronWrite · `CronTrigger`→CronWrite · `CronRuns`→CronRead · `CronPause`→CronWrite · `CronSuggestions`→CronRead · `CronAcceptSuggestion`→CronWrite · `CronDismissSuggestion`→CronWrite

**routing** (`serve_routing`): `RoutingListChats`→RoutingRead · `RoutingGet`→RoutingRead · `RoutingSet`→RoutingWrite · `RoutingBindChat`→RoutingWrite · `RoutingUnbindChat`→RoutingWrite · `TransportRooms`→RoutingRead · `TransportAdapters`→RoutingRead · `TransportInstances`→RoutingRead

**messaging** (`serve_messaging`): `ConvList`→MessagingRead · `ConvGet`→MessagingRead · `ConvCreateDetails`→MessagingRead · `ConvCreate`→MessagingWrite · `ConvJoinDetails`→MessagingRead · `ConvJoin`→MessagingWrite · `ConvLeave`→MessagingWrite · `ConvSend`→MessagingWrite · `ConvSetTopic`→MessagingWrite · `ConvSetTitle`→MessagingWrite · `ConvSetDescription`→MessagingWrite · `ConvDelete`→MessagingWrite · `ConvHistory`→MessagingRead · `MemberInvite`→MessagingWrite · `MemberRemove`→MessagingWrite · `MemberBan`→MessagingWrite · `MemberSetRole`→MessagingWrite · `ContactGetProfile`→MessagingRead · `ContactSetAlias`→MessagingWrite · `ContactActionMenu`→MessagingRead · `DirectorySearch`→MessagingRead

**registry** (`serve_registry`): `AcpDiscover`→RegistryWrite *(reviewable, fork #4)* · `AcpCatalog`→RegistryRead · `AcpRegister`→RegistryWrite · `AcpRemove`→RegistryWrite · `ProviderList`→RegistryRead · `ProviderRegister`→RegistryWrite · `ToolList`→RegistryRead · `ToolRegister`→RegistryWrite · `CommandList`→RegistryRead · `CommandInvoke`→RegistryRead *(coarse floor; command-level `min_access` does fine gating — fork #3)* · `ConfigGet`→RegistryRead · `ConfigSet`→RegistryWrite

**fs** (`serve_fs`): `FsRoots`→FsRead · `FsList`→FsRead · `FsStat`→FsRead · `FsRead`→FsRead · `FsWrite`→FsWrite · `FsSearch`→FsRead · `FsWatchPoll`→FsRead · `BlobPut`→FsWrite · `BlobGet`→FsRead · `BlobStat`→FsRead · `FsWriteFromBlob`→FsWrite

> Note: **no v2 `ApiRequest` variant maps to `AccessAdmin`** — the admin `AccessControl*`/`ResourceGrant*` DTOs were deferred to Track D. So `required_capability` never returns `AccessAdmin` today; Track D's `serve_access` variants will add those arms. The "Operator no AccessAdmin" matrix guard is therefore asserted at the `Principal` level here (see tests).

---

## 6. Test list (the plan's safety guards)

All in-crate (`#[cfg(test)]`); the concurrency test uses `#[tokio::test(flavor = "multi_thread", worker_threads = 2)]`.

**`request_context.rs`**
1. `default_context_denies` — outside any scope, `current_principal()` is `None` (⇒ no caps ⇒ deny).
2. `scope_binds_then_resets_to_deny` — inside `with_request_context`, `current_principal()` is `Some`; after the scope returns, it is `None` again (re-entered/dropped scope resets to deny).
3. `concurrent_tasks_are_isolated` — two spawned tasks, each in its own scope with a distinct principal, interleaved via `tokio::yield_now`/barrier; each only ever observes its own principal; the spawner (no scope) observes `None`.
4. `system_principal_is_full_local_trust` — `RequestContext::system()`: principal `.has(c)` for **every** `Capability` (iterate an explicit list incl. `AccessAdmin`, `SessionWrite`, `ControlWrite`…); `auth_method == Some(AuthMethod::LocalTrust)`; roles `== [Role::Admin]`.
5. `system_is_the_only_full_trust_constructor` — assert `authenticated(Principal::from_roles("u","x",[Role::User]), None)` does **not** hold `AccessAdmin`; documents that full trust comes only from `system()` (the "only intended sites" contract is enforced by it being the sole constructor that injects `Role::Admin`).

**`authz.rs`**
6. `exhaustiveness_is_compile_time` — doc-test/comment: `required_capability` has no `_` arm, so a new `ApiRequest` variant fails the build. Backed by `representative_mapping_per_group` (below) so the intent is also runtime-checked.
7. `unauthenticated_request_is_rejected` — with no scope, `authorize(&req)` is `Err(ApiError::Unauthenticated(_))` for a sample from each surface.
8. `authenticated_missing_capability_is_forbidden` — inside a `Viewer` scope, `authorize(&ApiRequest::Submit{..})` is `Err(ApiError::Forbidden(_))`.
9. `representative_mapping_per_group` — `required_capability` equals the expected cap for one variant per `serve_*` group (12+ asserts) — pins the table.
10. `role_matrix_viewer` — Viewer: every representative `*Write` variant ⇒ `Forbidden`; every representative read ⇒ `Ok`.
11. `role_matrix_user` — User: session/profile/credential/cron/messaging/fs **writes** ⇒ `Ok`; operator-tier ops (`Assign`/`Cancel` = ControlWrite, `Pause` = FleetWrite, `ModelDownload` = ModelsWrite, `RoutingSet` = RoutingWrite, `ConfigSet` = RegistryWrite) ⇒ `Forbidden`.
12. `role_matrix_operator` — Operator: the above operator-tier ops ⇒ `Ok`; and (Principal-level) `!Principal::from_roles("o","op",[Role::Operator]).has(Capability::AccessAdmin)` (the "Operator no AccessAdmin" guard, since no ApiRequest needs `AccessAdmin` yet).
13. `role_matrix_admin` — Admin passes every representative variant.

**`commands.rs`** (inversion guards; replace the existing `caller_access(None) == Admin` assert at L465)
14. `caller_access_without_principal_is_user_floor` — no scope ⇒ `caller_access(None) == CommandAccess::User` (the inverted polarity).
15. `caller_access_with_system_principal_is_admin` — inside `with_request_context(RequestContext::system())` ⇒ `Admin`.
16. `caller_access_with_user_principal_is_user_floor` — inside a `Role::User` scope ⇒ `User`.
17. `caller_access_with_operator_principal_is_admin` — inside a `Role::Operator` scope (has `ControlWrite`) ⇒ `Admin`.

---

## 7. Cross-track interface conformance (Track B / Auth 3)

| Track B assumed | This plan provides | Match? |
|---|---|---|
| `async with_request_context(ctx, fut)` | `pub async fn with_request_context<F: Future>(ctx, fut) -> F::Output` | ✅ |
| `current_principal()` | `pub fn current_principal() -> Option<Principal>` | ✅ |
| `authorize(&ApiRequest) -> Result<(),ApiError>` (reads task-local) | identical | ✅ |
| `RequestContext { principal: daemon_auth::Principal, origin: Option<Origin> }` | struct has those two fields **plus** `conn_id`/`auth_method`; use `RequestContext::authenticated(principal, origin)` / `system()` to construct | ⚠️ fork #1/#2 |
| Track B builds the local-trust `Principal`; A consumes | A also offers `RequestContext::system()` (full Admin local-trust) for the in-process/Unix path | ✅ (+ convenience) |

---

## 8. Design forks / blockers (coordinator, please confirm)

1. **`principal: Principal` (non-`Option`) vs the brief's `Option<Principal>`.** I recommend the
   **non-optional** field to match Track B's assumed shape and to keep fail-closed unambiguous:
   `current_principal()` is `None` *iff* no scope is active, and a scope can only be entered with a
   concrete principal (authenticated or `system()`). There is no "context present but principal
   absent" middle state to reason about. (If the coordinator prefers the literal `Option`, the only
   change is the field type + `current_principal` body; the gate is unaffected.)
2. **Extra `conn_id`/`auth_method` fields** (required by the brief, absent from Track B's two-field
   shape). Kept, but construction is via `RequestContext::authenticated(principal, origin)` /
   `system()` (+ `with_*` builders), so Track B never writes a struct literal. Confirm Track B uses
   the constructor (a bare `RequestContext { principal, origin }` literal would not compile).
3. **`CommandInvoke` → `RegistryRead` (coarse floor).** A command can be anything from `help` to a
   mutating op, so the per-variant gate is only a floor; the command catalog's own `min_access`
   (now principal-driven via the inverted `caller_access`) does the fine gating. Alternative:
   per-command capability mapping — out of scope for Auth 2 (revisit with Track D / audit).
4. **`AcpDiscover` → `RegistryWrite`.** It triggers an active scan that persists discoveries, so the
   conservative (fail-closed-leaning) choice is the write cap. Flag if discovery should be readable.
5. **`SessionUpdateMeta` → `SessionWrite`** (owner renames/pins/archives their own session). Could be
   `ControlWrite` if treated as an operator roster action; chosen `SessionWrite` so a normal user can
   curate their own roster.
6. **`Assign`/`Cancel`/`Rewind`/`CheckpointRewind`/`ApprovalDecide` → `ControlWrite`** (per the
   `ControlWrite` docstring "assign, cancel, approvals, rewind"). Consequence: a plain `User` cannot
   drive the **durable** lifecycle of even their own session via the control plane; that's an
   operator action (or via `SessionControlAny`, Track C). Confirm this is the intended product
   boundary (vs. mapping session-targeted lifecycle ops to `SessionWrite` + ownership).
7. **New dependency edges**: `daemon-host → daemon-auth` (+ a workspace-root
   `[workspace.dependencies]` entry). No new edge into `daemon-api`/`daemon-auth` themselves, so the
   layering invariant holds.
8. **`Forbidden` message** uses `{need:?}` (Rust enum name, e.g. `SessionWrite`). If audit/clients
   need the stable snake_case wire name, add a `Capability::as_str()` to `daemon-auth` (small,
   additive) — flag if wanted now.

---

## 9. Done-criteria for Phase 2 (implementation, after validation)

- `nix develop --command just build-all` and the `daemon-host` test suite green (incl. the 17 tests
  above); the `daemon-conformance` suite still green (no routing/wire change).
- `nix develop --command just lint` (rustfmt + clippy `-D warnings`) and `just deny` clean.
- No edits outside this worktree; no wire/CDDL change; `git` tree limited to the 6 files in §3.
