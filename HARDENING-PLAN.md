# HARDENING-PLAN — Phase 2 / Cluster F: revocation epoch (`secret-epoch`)

Worktree: `/home/j/experiments/daemon-worktrees/secret-epoch` (branch `hardening/secret-epoch`,
off `hardening/integration`). **Phase 1 = plan only. No source touched yet.**

## 1. The gap (what is broken today)

Two distinct "revoke the store, but the live thing keeps working" bugs. Both are the OpenClaw
class "the credential was withdrawn but the in-memory handle outlived it".

### Gap A — live mux connection outlives principal revocation
When an authenticated mux connection completes SASL, `serve_mux` snapshots the resolved
`Principal` into `ConnAuth::Authenticated { principal, method }`
(`crates/substrate/daemon-host/src/socket.rs:293-304`, set at `:509-517` / `:548-557`
/ `:602-606`) and **re-uses that snapshot for the entire connection lifetime**. Every `Call`
(`:628-659`) / `Open` (`:660-714`) rebuilds a `RequestContext` from the snapshot and dispatches.

`session_revoke` / `user_disable` / `user_set_roles` / `user_set_password`
(`crates/substrate/daemon-host/src/node_api/access.rs:188-196`, `:129-140`, `:142-154`,
`:156-167`) only mutate the **store** (`revoke_user_sessions` deletes `auth_sessions` rows;
`set_roles_guarded` rewrites roles). That invalidates the *reconnect fast-path* (`AuthResume`
→ `principal_for_token`, `authn.rs:294`) but does **nothing** to a connection already open:
it keeps issuing `Call`/`Open` with the old identity and old capability set indefinitely.
A live `Subscribe` pump (`socket.rs:732-827`) keeps streaming another owner's transcript after
the operator revoked the session — the exact cross-owner-read class Wave 1 closed for the
*initial* gate but not for post-revocation.

### Gap B — cached credential lease outlives `credential_remove`
`credential_remove(profile)` (`node_api/cred_auth.rs:25-33`) calls only `store.remove(profile)`.
The credential authority that mints/serves leases (`daemon-credentials/src/authority.rs`) is a
separate object cached per-profile in `MultiProfileStoreBroker.authorities`
(`daemon-host/src/credentials.rs:183-246`) and is never told. Consequences:
- A `Proxied` lease's real key is retained in `authority.proxied` (`authority.rs:70`, populated
  at `:167-172`); `use_capability` (`authority.rs:198-234`) keeps resolving it until TTL.
- Any already-minted `Bearer`/`Native` lease keeps resolving via `use_capability` (embedded
  secret at `authority.rs:221-224`) until its TTL (`60_000` ms on the node path,
  `bins/daemon/src/main.rs:2864`).

Note (scoped honestly): the node's engine acquires a **fresh** lease per model call
(`daemon-core/src/engine.rs:445-455`) and reads `lease.secret` directly for `Bearer` (never
calling `use_capability`), and `PooledStoreCredentialSource.provision`
(`credstore.rs:334-360`) reads the store fresh each acquire — so after `remove` the *next*
model call no longer sees the removed key (it falls back). The residual, genuinely-exploitable
reuse is: (1) `Proxied` retained keys, (2) any holder that resolves an already-minted lease via
`use_capability` (relay/cut holders, `cut.rs:875-900`), (3) an in-flight call (unrevocable,
bounded — out of scope). Part B closes (1) and (2) with an authority epoch; (3) is documented
as residual.

## 2. Epoch data model

Two **independent** epoch registries, each per-subject, both in-memory (a live connection / live
lease is a this-process artifact; the durable store is already the source of truth for
reconnect). Neither is a global counter (a global would tear down unaffected principals).

### Part A — principal/connection epoch (`SessionRevocations`, new, in `daemon-host`)
Keyed by **`user_id`** (per-principal). Lives in `daemon-host` (not `daemon-auth`) so it can
carry a tokio `Notify` for *synchronous* wake without pulling async into the pure `daemon-auth`
crate.

```rust
// new: crates/substrate/daemon-host/src/revocation.rs
pub struct SessionRevocations { users: Mutex<HashMap<String, Arc<UserRevocation>>> }
struct UserRevocation { epoch: AtomicU64, notify: Notify }

impl SessionRevocations {
    pub fn new() -> Arc<Self>;
    fn cell(&self, user_id: &str) -> Arc<UserRevocation>;         // get-or-create
    pub fn guard(&self, user_id: &str) -> RevocationGuard;        // captures current epoch
    pub fn revoke(&self, user_id: &str);                          // epoch += 1; notify_waiters()
}
#[derive(Clone)]
pub struct RevocationGuard { cell: Arc<UserRevocation>, at: u64 } // Arc + u64: cheap to clone
impl RevocationGuard {
    pub fn is_revoked(&self) -> bool { self.cell.epoch.load(Acquire) != self.at }
    pub async fn revoked(&self) { /* Notified-before-recheck loop, lost-wakeup-safe */ }
}
```

Sharing (zero new construction risk — one `Arc` created in `main.rs`, wired both sides):
- **Capture**: injected into the transport via `Authenticator` (`Authenticator::with_revocations`
  + `revocations()` accessor). `serve_mux` reaches it through `AuthMode::Required { auth, .. }`
  and calls `auth.revocations().guard(&principal.user_id)` at the instant it sets
  `ConnAuth::Authenticated`. `AuthMode::LocalSystem` (unix local-trust / FFI) captures **no**
  guard — local trust is deliberately non-revocable.
- **Bump**: injected into `NodeApiImpl` (`with_revocations`). The four handlers call
  `revocations.revoke(user_id)` right after the store mutation commits.
- **Check**: `ConnAuth::Authenticated { principal, method, guard: Option<RevocationGuard> }`.
  Enforced in `serve_mux` (see §3) and in each spawned pump (guard clone).

Because TLS (`tls.rs:173`), WS (`ws.rs:125`) and the web-front `/ws` (`web.rs:611`) all funnel
through the *same* `serve_mux`, one enforcement point covers every authenticated mux carrier.

### Part B — credential/lease epoch (on `CredentialAuthority`)
Keyed per **authority** (one authority == one profile in `MultiProfileStoreBroker`).

- `CapabilityLease` gains `pub epoch: u64` (`daemon-common/src/lib.rs:883-898`), stamped at mint,
  **covered by the signature** (added to `capability_digest`, `daemon-credentials/src/capability.rs:47-56`)
  so a relay cannot re-stamp it.
- `CredentialAuthority` gains `epoch: AtomicU64` (`authority.rs:62-74`), starting `0`.
  - `acquire` (`:127-193`) stamps `epoch: self.epoch.load(Acquire)` into the lease.
  - `use_capability` (`:198-234`) rejects `lease.epoch != self.epoch.load(Acquire)` with
    `CredError::Unavailable` (checked after signature/expiry, before returning any secret).
  - new `revoke_all(&self, ctx)`: `epoch.fetch_add(1, AcqRel)`, clear `proxied`, push audit
    `Revoke`.
- `MultiProfileStoreBroker` gains `revoke_profile(&self, profile: &str)` (`credentials.rs`):
  look up (do not create) the authority under the `authorities` lock, clone-and-drop, call
  `revoke_all`.
- Injected into `NodeApiImpl` as `Option<Arc<dyn CredentialRevoker>>` (new trait
  `CredentialRevoker { fn revoke_profile(&self, profile: &str); }`, impl'd by
  `MultiProfileStoreBroker`), wired in `main.rs` from the existing `owner_broker`.
- `EmbeddedCredentialPool` (`daemon-core/src/credentials.rs:185-193`) and the engine test double
  (`engine/tests.rs:183`) construct leases with `epoch: 0` (standalone L1 has no remote
  revocation authority; epoch is inert there — the engine reads `lease.secret` directly).

## 3. How each operation forces synchronous teardown

### `serve_mux` enforcement (Part A, `socket.rs`)
1. `ConnAuth::Authenticated` grows a `guard: Option<RevocationGuard>` captured at auth success
   (Required mode) / `None` (LocalSystem).
2. Main loop frame read (`:466-477`): when a guard is present, **race** `read_frame` against
   `guard.revoked()` in a `tokio::select!`. On the revoked branch → drain+abort all `streams`
   (`for (_,h) in streams.drain() { h.abort() }`) and `break` (drop `tx` → writer ends → socket
   closes). This makes an *idle* authenticated connection close promptly on revoke.
3. `Call` (`:628`) / `Open` (`:660`): before spawn, `if let Some(g)=&guard { if g.is_revoked()`
   → reply `Unauthenticated` (Call) / `End{Unauthenticated}` (Open), then `break`. Closes the
   TOCTOU window where a frame arrived just as revoke fired.
4. `spawn_stream` (`:732`) takes a cloned `guard`; the pump loops (`pump_session_log:792`,
   `pump_node_events:844`) add a `guard.revoked()` arm to their `select!` and check
   `is_revoked()` at loop top → send `End{Some(Unauthenticated)}` and return. Belt-and-suspenders
   with (2)'s abort; matches the plan's "live mux pumps holding a stale epoch are closed".

Result: after `revocations.revoke(user_id)` returns, every live connection for that user (a)
cannot issue a new `Call`/`Open` (denied + closed), (b) has its live `Subscribe` pumps torn down,
(c) has its socket closed. New logins after the bump capture the new epoch → unaffected.

### The four (+2) trigger operations
| op | file:line | store effect (unchanged) | added |
|---|---|---|---|
| `session_revoke` | `access.rs:188` | `revoke_user_sessions` | `revocations.revoke(user_id)` |
| `user_set_roles` | `access.rs:142` | `set_roles_guarded` | `revocations.revoke(user_id)` |
| `user_set_password` | `access.rs:156` | `set_password`+`revoke_user_sessions` | `revocations.revoke(user_id)` |
| `user_disable`(→true) | `access.rs:129` | `set_disabled_guarded` (revokes sessions) | `revocations.revoke(user_id)` (included: disable revokes sessions, so a live conn must die) |
| `credential_remove` | `cred_auth.rs:25` | `store.remove(profile)` | `credential_revoker.revoke_profile(profile)` |
| `credential_set` | `cred_auth.rs:8` | `store.set` | `credential_revoker.revoke_profile(profile)` (included: a rotation must invalidate leases minted against the old key) |

`user_disable`/`credential_set` are **APPROVED** as bump triggers (same vulnerability class,
cheap, broader revocation coverage is the correct posture). Full trigger set: `session_revoke`,
`credential_remove`, `user_disable`, `credential_set`, role change (`user_set_roles`), password
change (`user_set_password`).

## 4. Concurrency / deadlock (this touches live async pumps)

**Lock inventory & ordering (no nesting → no deadlock):**
- `SessionRevocations.users` (`Mutex<HashMap>`) — a **leaf** lock, held only for the
  get-or-create of an `Arc<UserRevocation>`, never across an `.await` and never while any other
  lock is held. `revoke()` = take `users` briefly → clone cell → drop → `fetch_add` (atomic) →
  `notify_waiters()` (lock-free).
- `AuthStore.conn` (`Mutex<Connection>`) — unchanged. The epoch bump happens in the **node_api
  handler**, strictly *after* the store method returns (its `conn` guard already dropped). Order
  per handler: `conn` (store op) → drop → `users` (bump). Never nested.
- `serve_mux` per-connection task holds no shared lock across awaits. `streams: HashMap<u64,
  AbortHandle>` is task-local. `handle.abort()` and dropping `tx` are non-blocking.
- `Notify` wake pattern uses the canonical "create `notified()` future, then re-check the atomic"
  loop so a bump racing a subscribe cannot be lost; the `AtomicU64` epoch is the authoritative
  source of truth (Notify is only a prompt-wake optimization) — an idle stream that somehow
  missed the wake still tears down at its next keepalive tick (`STREAM_KEEPALIVE=20s`,
  `socket.rs:46`) via the top-of-loop `is_revoked()` check.
- `CredentialAuthority`: `revoke_all` takes `proxied` (clear, scoped-drop) then `audit` (push) —
  same independent locks the existing `revoke` (`:243-254`) already takes, no new nesting; the
  epoch is an atomic. `MultiProfileStoreBroker::revoke_profile` clones the `Arc<CredentialAuthority>`
  out from under the `authorities` lock and drops it before calling `revoke_all` (so the
  `authorities` lock is never held across the authority's `proxied`/`audit` locks).
- Memory ordering: `Acquire`/`Release`(`AcqRel`) on the epoch atomics — a connection/lease that
  observes the store mutation must observe the bump; the handler orders store-op-then-bump, and
  the reader loads the epoch on each check.

## 5. Tests — added FIRST, confirmed red before the fix

Because both parts introduce new plumbing, "red pre-fix" = **plumbing landed, enforcement
omitted**: add the data model + wiring, write the test asserting the *secure* outcome, run and
confirm it FAILS (the pre-enforcement build still serves the revoked connection / lease), then add
the enforcement and confirm it passes. Each step's failing/passing run is captured in the commit.

### Part A
- `daemon-host` unit (`revocation.rs` `#[cfg(test)]`): `guard.is_revoked()` flips after `revoke`;
  distinct users are independent; `revoked()` completes after a concurrent `revoke`.
- `daemon-conformance` new `src/node/revocation_transport.rs` (harness pattern from
  `auth_transport.rs`), wiring a shared `Arc<SessionRevocations>` into `Authenticator` + node:
  1. **`revoked_session_tears_down_live_connection`**: SCRAM-auth operator over
     `serve_api_unix_authenticated`; `Health` Call succeeds; `revocations.revoke(operator_id)`
     (the seam `session_revoke` drives); assert the next `client.call(Health)` errors (connection
     closed) or returns `Unauthenticated`. *Pre-fix (no serve-loop check): the second Call
     succeeds → red.*
  2. **`revoked_session_ends_live_subscribe_stream`**: operator opens a `Subscribe` stream, reads
     one frame; `revoke`; assert `client.next()` yields `End`/EOF. *Pre-fix: the pump keeps
     streaming → red.*
  3. **`session_revoke_op_bumps_revocation`** (handler→registry wiring): build the node with the
     shared registry (via a `&self` `set_revocations` setter, mirroring `set_commands`), call
     `as_system(node.session_revoke(user_id))`, assert the user's epoch advanced. Repeat for
     `user_set_roles` / `user_set_password` / `user_disable`.
  4. Regression: a **different** user's live connection is *not* torn down by revoking user X
     (per-principal, not global).

### Part B
- `daemon-credentials` unit (`authority.rs` tests): `bearer_lease_use_fails_after_revoke_all`
  (use ok → `revoke_all` → `use_capability` == `Unavailable`); `proxied_key_dropped_on_revoke_all`
  (proxied resolves → `revoke_all` → `Unavailable` and `proxied` empty);
  `epoch_is_signed` (tampering `lease.epoch` → `BadSignature`).
- `daemon-host` unit (`credentials.rs` tests): `broker_revoke_profile_invalidates_outstanding_lease`
  (acquire → use ok → `revoke_profile` → use `Unavailable`; a *fresh* acquire after revoke
  succeeds under the new epoch); revoking profile A does not disturb profile B.
- Existing `daemon-conformance/src/credentials.rs` behavior tests must still pass unchanged (the
  new `epoch` field is additive; both ends same build).

## 6. Wire-format impact

- **`CapabilityLease` gains `epoch: u64`.** This type is **NOT** in `daemon-api.cddl` (verified:
  no match for `CapabilityLease`/`cap_id`/`CredScope` in the only `.cddl`,
  `crates/contracts/daemon-api/daemon-api.cddl`). It is an internal **credential-cut-protocol**
  type (`daemon-common`), serialized transparently via serde-CBOR inside `CredCall`/`CredReplyBody`
  (`cut.rs:137-150`) — both cut ends are the same node build, so the field add is
  backward-compatible in practice and needs **no** CDDL edit.
- It derives `arbitrary::Arbitrary` under the `arbitrary` feature (`daemon-common/src/lib.rs:881`);
  `u64` is `Arbitrary`, so the derive keeps compiling.
- `daemon-api` is untouched, so `cargo test -p daemon-api --features arbitrary` should pass
  unchanged — I will still run it (per gate policy) to prove no daemon-api wire drift, and
  explicitly confirm no `daemon-api.cddl` change is required.
- No new `ApiRequest`/`ApiResponse` variants or fields — the six trigger ops already exist.
  `authz.rs::required_capability` is unchanged (all six ops already mapped:
  `SessionRevoke`/`UserSetRoles`/`UserSetPassword`/`UserDisable` → `AccessAdmin`,
  `CredentialRemove`/`CredentialSet` → `CredentialWrite`). No conformance ownership-matrix change.

## 7. Residual coverage (documented, not closed here)

- **In-flight model call**: a `Bearer` lease's secret already threaded into an in-flight HTTP
  request (`engine.rs:452`) cannot be pulled mid-flight; bounded by the call duration / lease TTL.
- **`daemon-http` SSE/GET log routes** (`daemon-http`): a separate transport from the mux; its
  principal handling was Wave 1's concern. Not a `serve_mux` connection, so not covered by Part A;
  noted for a follow-up (Phase 4 ingress governor is the natural home).
- **`credential_set` fallback surprise**: after `credential_remove`, a fresh acquire falls back to
  the launch `fallback_key` if configured (`credstore.rs:349-353`) — pre-existing zero-config
  behavior, not a revocation regression (the *removed* key is not re-issued). Out of scope.
- **Local-trust connections** (unix `serve_api_unix`, FFI): deliberately non-revocable (no guard);
  the local-trust posture is an accepted deployment choice (superproject `AGENTS.md`).

## 8. Exact gate commands (from worktree root, tails shown)

```
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo test -p daemon-api --features arbitrary   # wire-type gate (expect no drift)
```
Targeted while iterating: `-p daemon-credentials`, `-p daemon-host`, `-p daemon-common`,
`-p daemon-conformance`.

Known pre-existing timing flakes (do **not** chase; this track perturbs the mux/session
lifecycle, so prove no regression). Baseline on clean master: ~2/6 isolation-failure rate, 30s
timeout signature. After the change, stress each ~6× in isolation and report the ratio vs
baseline:
```
for i in $(seq 1 6); do nix develop --command cargo test -p daemon-conformance \
  node::detached_delegation::detached_notice_reaches_a_parked_durable_parent \
  node::detached_delegation::detached_fanout_materializes_distinct_children \
  node::process_notify::injected_input_reaches_a_parked_durable_session_via_the_store_seam \
  -- --exact --test-threads=1 --no-fail-fast; done
```

## 9. File inventory (edits, by crate)

- `crates/contracts/daemon-common/src/lib.rs` — `CapabilityLease.epoch: u64` (`:883-898`).
- `crates/substrate/daemon-credentials/src/authority.rs` — `epoch: AtomicU64` field; stamp in
  `acquire` (`:175-183`); check in `use_capability` (`:203-206`); new `revoke_all` (near `:243`).
- `crates/substrate/daemon-credentials/src/capability.rs` — add `epoch` to `capability_digest`
  (`:47-56`).
- `crates/engine/daemon-core/src/credentials.rs` — `epoch: 0` in the pool lease (`:185-193`).
- `crates/engine/daemon-core/src/engine/tests.rs` — `epoch: 0` in the test-double lease (`:183`).
- `crates/substrate/daemon-host/src/revocation.rs` — **new** `SessionRevocations` + tests; export
  in `lib.rs`.
- `crates/substrate/daemon-host/src/authn.rs` — `Authenticator.revocations` +
  `with_revocations`/`revocations()` (`:173-207`).
- `crates/substrate/daemon-host/src/socket.rs` — `ConnAuth::Authenticated { .., guard }`
  (`:293-304`); capture at auth-success sites (`:446-454` none, `:509-517`, `:548-557`,
  `:602-606`); main-loop revoke race + Call/Open pre-dispatch check (`:466-722`);
  `spawn_stream`/pump guard arm (`:732-868`).
- `crates/substrate/daemon-host/src/credentials.rs` — `CredentialRevoker` trait +
  `MultiProfileStoreBroker::revoke_profile` (`:183-287`).
- `crates/substrate/daemon-host/src/node_api.rs` — `NodeApiImpl.revocations` +
  `.credential_revoker` fields (`:293-433`).
- `crates/substrate/daemon-host/src/node_api/assembly.rs` — `with_revocations`/`set_revocations`,
  `with_credential_revoker` builders; defaults `None` (`:44-85`).
- `crates/substrate/daemon-host/src/node_api/access.rs` — `revocations.revoke` in
  `session_revoke`/`user_set_roles`/`user_set_password`/`user_disable` (`:129-196`).
- `crates/substrate/daemon-host/src/node_api/cred_auth.rs` — `revoke_profile` in
  `credential_remove`/`credential_set` (`:8-33`).
- `bins/daemon/src/main.rs` — create one `Arc<SessionRevocations>`; `Authenticator::with_revocations`
  (`:2500`), `node.with_revocations` (`:2492-2496`), `node.with_credential_revoker(owner_broker)`
  (`:2077-2078`, `:2492-2496`).
- `tests/daemon-conformance/src/node/revocation_transport.rs` — **new** (register in the node
  module list).

## 10. Review decisions (APPROVED)

1. Scope: `user_disable` + `credential_set` included as bump triggers (full set above).
2. Wire: `CapabilityLease.epoch` enters the signed digest, not the CDDL; run
   `cargo test -p daemon-api --features arbitrary` to prove zero drift.
3. Concurrency: leaf registry lock; bump strictly AFTER the store `conn` guard drops; no nesting;
   local-trust non-revocable; the frame-read↔`guard.revoked()` race and pump teardown must not
   hang. If the bump can be observed while a store lock is held → STOP.
4. Residuals accepted (in-flight model call, daemon-http SSE routes) — tracked as follow-on, not
   expanded here.
5. Only Wave 2 track touching `socket.rs` / mux lifecycle — keep hunks minimal.
