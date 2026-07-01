# daemon access control (authentication + authorization)

Status: shipped (Auth 1–7). This spec is the authoritative description of the node's identity,
authentication, and authorization model. It is a distinct axis from the release version (the
`VERSION` files) and the on-disk schema versions (`PRAGMA user_version`); it governs **who** may
connect and **what** they may do.

The model is built bottom-up and is fully integrated in the node:

- the foundation crate `daemon-auth` — the capability/role model + the SQLite identity store
  (users, Argon2id password hashes, RFC 5802 SCRAM material, opaque session tokens, EXTERNAL
  cert→user mappings, the reserved `resource_grants` table);
- the node authenticator (`daemon-host::authn`) — an `rsasl` SASL state machine over the store;
- the per-request capability gate + task-local request context (`daemon-host::authz` /
  `request_context`);
- session/orchestration-tree ownership (stamped owner, owner-scoped enumeration and control);
- the admin AccessControl API (`daemon-host::node_api::access`) over the wire;
- the authn/authz audit chain (`daemon-host::auth_audit`) on the verifiable journal.

The integrated path — handshake → SASL auth → request context → capability gate → ownership →
dispatch — is proven end to end by the `daemon-conformance` `node::` suite (positive e2e + the
fail-closed negative suite, over both the Unix socket and a real `rustls` TLS/TCP connection).

## 1. Threat model

- **Networked from day one.** Clients (the GUI, the TUI, the CLI, future agents) may connect from
  other machines, so any non-local transport is assumed hostile: it carries TLS, and credentials are
  never sent in cleartext.
- **The node is the sole authority.** All enforcement is server-side. Anything a client is told about
  its own roles/capabilities (`PrincipalView` on `AuthOk`) is advisory, for UI gating only.
- **Fail-closed.** The absence or ambiguity of identity denies access; it never implies admin. A new
  API operation with no capability mapping must fail the build/tests, not silently pass.
- **Local trust is explicit.** The in-process FFI and the local Unix socket may run as a configured
  system principal (`[api].local_trust`), but that is a deliberate, tested decision — never an
  accident of "no auth configured".

## 2. Authentication mechanisms

Authentication is a SASL-style exchange carried in the L0 mux envelope (`AuthStart`/`AuthStep`/
`AuthResume` -> `AuthChallenge`/`AuthOk`/`AuthError`), independent of the request transport. The
server advertises offered mechanisms in its `Hello.auth_mechanisms`.

| Mechanism | Use | Policy |
|---|---|---|
| `SCRAM-SHA-256` | Username/password without sending the password | Preferred for interactive login |
| `PLAIN` | Username/password verified against an Argon2id PHC hash | Allowed **only over TLS** |
| `EXTERNAL` | mTLS client certificate -> user (verified leaf-cert SHA-256 fingerprint) | Machine/node auth; **mTLS only** |
| session token | Opaque server-issued token via `AuthResume` | Reconnect fast-path |
| OAUTHBEARER/OIDC, passkeys/WebAuthn, TOTP | SSO, hardware authenticators, 2FA | Deferred; mechanism + table seams reserved |

The server advertises `SCRAM-SHA-256` on every transport; `PLAIN` and `EXTERNAL` are advertised
**only over TLS** (PLAIN sends the password; EXTERNAL needs a client certificate). On a successful
exchange the authenticator mints an opaque session token — **only** on success, never on a challenge
or failure — and resolves the caller's `Principal`; the token rides `AuthOk` and is presented on
reconnect via `AuthResume`.

`EXTERNAL` is functional: a verified client-cert fingerprint is mapped to a user via the
`external_identities` table (`AuthStore::external_identity` / `set_external_identity`); an unmapped
fingerprint (or a disabled mapped user) denies (fail-closed). Only the *admin enrollment op* (a wire
surface to register a fingerprint→user mapping) is deferred to a later track — enrollment today is the
store-level writer.

Password storage is Argon2id (OWASP baseline). Session tokens are random, stored only as a SHA-256
hash, server-side and revocable (never JWTs in the DB). See `daemon-auth`.

SCRAM-SHA-256 stores RFC 5802 derived material (`salt`/`iterations`/`StoredKey`/`ServerKey`),
derived from the password whenever it is set (`create_user` / `set_password`). This material cannot
be back-derived from the existing Argon2id hash, so **a user provisioned before SCRAM derivation was
introduced has no SCRAM material until their password is next set**: that user cannot use
`SCRAM-SHA-256` until an admin (or the user) re-sets the password, but `PLAIN`-over-TLS keeps working
in the meantime (it verifies the Argon2id hash directly). To avoid an account-probing oracle, the
authenticator serves a deterministic *decoy* SCRAM credential for an unknown, disabled, or
SCRAM-material-less user, so the exchange fails at proof verification exactly like a wrong password.

## 3. Authorization: roles and capabilities

Authorization is two-step: a coarse per-request **capability** gate, plus a per-resource
**ownership** check for sessions. Capabilities are aligned to the node's API operation categories.

Roles form a monotonic ladder (each a superset of the previous), admin-assignable per user:

| Role | Grants |
|---|---|
| `viewer` | read-only over one's own surfaces (`*_read`) |
| `user` | + write over one's own resources (`session_write`, `profile_write`, ...) |
| `operator` | + node-wide visibility/control: `session_see_all`, `session_control_any`, `control_write`, `fleet_write`, `routing_write`, `registry_write` |
| `admin` | + `access_admin` (manage users/roles/sessions) |

The capability vocabulary and the role->capability mapping are defined in
[daemon-auth/src/capability.rs](../../crates/substrate/daemon-auth/src/capability.rs) (the single
source of truth; `PrincipalView` on the wire is its serialized mirror).

The per-request gate is `daemon-host::authz::required_capability(&ApiRequest)` → `authorize`. It is
one **exhaustive** match with NO wildcard arm: every `ApiRequest` variant declares its required
access (`Authenticated`, or a specific `Capability`), so a newly-added operation with no mapping
fails the build/tests rather than silently passing. `WhoAmI` requires only that *some* principal is
authenticated; the admin ops require `access_admin`; the rest map to their category capability. The
gate runs inside the request's task-local `RequestContext`; a `tokio::spawn`ed per-request task does
NOT inherit that scope, so the transport re-establishes it before the gate + dispatch — otherwise
`current_principal()` is `None` and the gate denies (fail-closed).

## 4. Transport policy: local trust vs networked

Identity is established per connection by the transport, never assumed:

- **In-process FFI / local Unix socket** — may run under `[api].local_trust`, which binds the
  explicit `RequestContext::system` principal (`SYSTEM_USERNAME`, full capabilities). This is a
  deliberate, named, audited principal — **not** admin-by-absence. Under local trust the socket
  advertises NO SASL mechanisms and runs no exchange. With `local_trust` disabled the Unix socket
  behaves exactly like the networked path below (it requires a SASL exchange).
- **Networked TCP** — always TLS, and always `AuthMode::Required`: TCP is never local-trusted. A
  connection must complete a SASL exchange ending in `AuthOk` before any `Call`/`Open`; a pre-auth
  request resolves to `ApiError::Unauthenticated` and the connection stays unelevated. The TLS layer
  (rustls, aws-lc-rs provider) always presents a server certificate; under mTLS it verifies the
  client certificate and captures its fingerprint for EXTERNAL.

The bare (non-multiplexed, no-`Hello`) one-shot protocol carries no handshake, so it is served only
under local trust; when auth is required it refuses with `Unauthenticated` (a networked client must
use the multiplexed SASL path).

## 5. Session and orchestration-tree ownership

- Every session carries an `owner` (the creating principal); delegated/background/cron children
  inherit their parent's/job's owner (the worker has no principal of its own).
- Roster/list enumeration, `session_get`/`session_search`, and the fleet `tree` are filtered to the
  caller's owned sessions/subtree unless the caller holds `session_see_all`. A peer's `session_get`
  returns `None` (no existence oracle), not a denial.
- Session-targeted operations (poll/respond/cancel/submit/handover/...) require the caller to own the
  session or hold `session_control_any`.
- A legacy `owner IS NULL` session (created by the trusted in-process path with no bound principal)
  is hidden from a non-operator peer and reachable only via the `session_see_all`/`session_control_any`
  overrides.
- `PartitionId` remains the placement/activation axis; ownership is the orthogonal per-user axis.

## 6. The admin AccessControl API

Admin user/role/session administration is the `AccessControlApi` surface (wire variants
`UserCreate`/`UserList`/`UserDisable`/`UserSetRoles`/`UserSetPassword`/`RoleList`/`SessionRevoke`/
`WhoAmI`, plus the reserved `ResourceGrant*`), implemented in
[node_api/access.rs](../../crates/substrate/daemon-host/src/node_api/access.rs):

- Every op except `WhoAmI` requires `access_admin`. The capability gate enforces this on the
  transport, and each handler **re-checks** it (defense in depth) so the in-process/FFI path and any
  future caller cannot reach an admin mutation without the capability. `WhoAmI` is allowed for any
  authenticated principal (it returns the caller's own `PrincipalView`).
- **Last-admin lockout** is enforced atomically by the store's guarded mutations
  (`set_disabled_guarded`/`set_roles_guarded`): the final administrator cannot be demoted or disabled.
- `UserSetPassword` re-derives SCRAM material (keeping PLAIN/SCRAM coherent) and revokes the user's
  sessions, so a reset forces re-login; `UserDisable`/`SessionRevoke` revoke tokens too.
- Unknown role strings are rejected (fail-closed: never silently dropped to "no role").

## 7. Reserved extension: per-resource grants (option B)

Fine-grained sharing (granting one capability over one specific session/profile/agent to one user)
is reserved but **not yet enforced**: the `resource_grants` table exists in the `daemon-auth` store,
and a future `grants_allow(principal, resource, capability)` hook slots between the capability gate
and the ownership check. The admin API reserves the `ResourceGrantCreate`/`ResourceGrantList`/
`ResourceGrantRevoke` variants, which currently return `ApiError::Unsupported`, so enabling sharing
later is purely additive — no wire/protocol change.

## 8. Audit

Authentication and authorization events — login success/failure, permission denials, user CRUD, role
changes, password reset, session revocation — are recorded by the shared `AuthAudit`
([auth_audit.rs](../../crates/substrate/daemon-host/src/auth_audit.rs)) onto a dedicated `node-auth`
stream of the verifiable journal ([journal.rs](../../crates/substrate/daemon-host/src/journal.rs)),
append-only and tamper-evident (each event seals a segment chaining onto the prior root; the chain
verifies against the trace signer's key). The same handle is shared by the transport (login + denial
events) and the node interface (admin events), so every auth event rides one chain.

Audit payloads carry identifiers and outcomes only (`user_id=…`, `username=…`, `method=…`,
`roles=[…]`, `op=…`) and **never** credential material: no passwords, no session tokens, no
PHC/SCRAM blobs. A denial records the payload-free op tag (the variant name), never the request body,
which may itself carry a credential (e.g. `UserCreate`).

## 9. First-admin bootstrap (empty-store seeding)

A fresh node has an empty identity store, so a networked/TLS operator would have no account to log
in as. The bootstrap seeds **exactly one `Admin` user, iff the users table is empty**, and is
idempotent — a second boot (any user already present) is a no-op that never re-seeds. The pure store
decision is `AuthStore::seed_first_admin_if_empty(AdminSeed)`
([daemon-auth/src/bootstrap.rs](../../crates/substrate/daemon-auth/src/bootstrap.rs)); environment
resolution + the one-time secret emission are the binary's responsibility
([bins/daemon/src/main.rs](../../bins/daemon/src/main.rs) `resolve_admin_seed` /
`seed_first_admin_if_empty` / `emit_generated_admin`), run once in `run_as_host` after the
`AuthStore` is bound.

- **Env-first.** With `DAEMON_ADMIN_USERNAME` set, the password is taken from `DAEMON_ADMIN_PASSWORD`
  or, failing that, from the file at `DAEMON_ADMIN_PASSWORD_FILE`. An empty/whitespace password is
  **refused** (the launch errors) — the node never seeds `admin`/`<blank>` or any password-less user.
- **Auto-generate otherwise.** With no `DAEMON_ADMIN_USERNAME`, the node mints a random
  `admin-<hex>` username and a strong random password.
- **One-time emission (the sole deliberate secret print).** The auto-generated password is emitted
  **exactly once**: to stderr and to a `0600` `first-admin-credentials.txt` under the data dir. It is
  never routed through `tracing` (so it stays out of structured logs/journald) and, unlike every
  other credential, this is the one documented exception to §8's "no credential material" rule — it
  never enters the audit journal. The operator-supplied (env) path emits nothing (the operator
  already knows the password); only the id is logged.
- **Interaction with `local_trust` (§4).** Under the default `local_trust=system` the local
  operator is already a full-trust admin over the Unix socket/FFI **without** any SASL exchange, so
  the seeded admin exists primarily to give a **networked/TLS** operator a real Admin identity to
  authenticate as (SCRAM/PLAIN) and to make the admin `AccessControl` API usable off the local path.
  Seeding never weakens the fail-closed gate: it adds an identity, it does not grant access by
  absence of one.

The over-the-wire proof that a seeded admin authenticates (SCRAM) and drives audited user CRUD is in
the `daemon-conformance` `node::positive_e2e` suite; the whole login→profile→credential→chat chain
(seeded admin → SCRAM → `CredentialSet` → `DaemonApi` profile → turn) is `node::live_agent_e2e`.
