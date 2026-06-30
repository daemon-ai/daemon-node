# daemon access control (authentication + authorization)

Status: in progress. This spec defines the node's identity, authentication, and authorization model.
It is a distinct axis from the release version (the `VERSION` files) and the on-disk schema versions
(`PRAGMA user_version`); it governs **who** may connect and **what** they may do.

The foundation crate `daemon-auth` (capability/role model + SQLite identity store) is implemented.
The wire contract for the authentication handshake is frozen (this increment). Remaining layers
(authenticator, authorization gate, ownership, admin API, clients) land in later increments; see the
`auth_*` plans.

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
| `EXTERNAL` | mTLS client certificate -> user (cert fingerprint mapping) | Machine/node auth |
| session token | Opaque server-issued token via `AuthResume` | Reconnect fast-path |
| OAUTHBEARER/OIDC, passkeys/WebAuthn, TOTP | SSO, hardware authenticators, 2FA | Deferred; mechanism + table seams reserved |

Password storage is Argon2id (OWASP baseline). Session tokens are random, stored only as a SHA-256
hash, server-side and revocable (never JWTs in the DB). See `daemon-auth`.

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

## 4. Session and orchestration-tree ownership

- Every session carries an `owner` (the creating principal); delegated/background/cron children
  inherit their parent's owner.
- Roster/list enumeration and the fleet tree are filtered to the caller's owned sessions/subtree
  unless the caller holds `session_see_all`.
- Session-targeted operations (poll/respond/cancel/handover/...) require the caller to own the
  session or hold `session_control_any`.
- `PartitionId` remains the placement/activation axis; ownership is the orthogonal per-user axis.

## 5. Reserved extension: per-resource grants (option B)

Fine-grained sharing (granting one capability over one specific session/profile/agent to one user)
is reserved but not yet enforced: the `resource_grants` table exists in the `daemon-auth` store, and
a future `grants_allow(principal, resource, capability)` hook slots between the capability gate and
the ownership check. The admin API reserves `ResourceGrant*` variants (returning `Unsupported` until
then), so enabling sharing is purely additive — no wire/protocol change.

## 6. Audit

Authentication and authorization events (login success/failure, user CRUD, role changes, permission
denials, session revocation) are recorded into the existing verifiable journal
([daemon-host/src/journal.rs](../../crates/substrate/daemon-host/src/journal.rs)), append-only and
tamper-evident. Audit payloads never contain credential material (no passwords, tokens, or PHC).
