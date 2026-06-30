<!--
SPDX-License-Identifier: MIT OR Apache-2.0
SPDX-FileCopyrightText: 2026 Jarrad Hope
-->

# Auth 3 — authentication + transport (Track B) implementation plan

Worker plan for the meta-plan node `auth3-authn-transport`
(`/home/j/.cursor/plans/auth_rollout_meta_plan_9682285d.plan.md`).

**Worktree discipline:** all work happens in
`/home/j/experiments/daemon-worktrees/auth3-authn-transport` (branch
`feature/auth3-authn-transport`, a worktree of the `daemon-node` Rust repo). Never touch
`/home/j/experiments/daemon/daemon-node` or sibling worktrees. All commands run through Nix:
`nix develop --command <cmd>` (no host tools). Gates before "done": `just lint`, `just deny`,
`just build-all`, plus the new per-crate tests.

This is **PHASE 1 (plan only)** — no source is modified until the coordinator validates.

---

## 0. Scope recap and the independence rule

Three deliverables, two of which land **independently first** and one that is the **convergence
step the coordinator sequences after Auth 2 merges**:

1. **rsasl-authenticator** (independent) — `crates/substrate/daemon-auth` gains the SCRAM material
   derivation + a transport-agnostic `Authenticator` built on `rsasl`'s `SessionCallback` over
   `AuthStore`. No transport, no Track-A types. Fully unit-testable in isolation.
2. **tls-listener** (independent) — `crates/substrate/daemon-host/src/socket.rs` gains
   `serve_api_tls_tcp` over `tokio-rustls`; `bins/daemon` gains `[api]` config + wiring. The TLS
   accept/handshake path is testable on its own (a TCP client doing a rustls handshake), and can be
   stubbed to require auth without yet consuming Track A.
3. **serve_mux integration** (CONVERGENCE — do last) — the handshake → auth → context → authorize →
   dispatch state machine. This consumes Track A's `with_request_context` / `current_principal` /
   `authorize` (Auth 2, a sibling worktree **not yet merged**). We build it against the *assumed*
   interfaces in §8, behind a thin local shim so it compiles and is unit-tested in this worktree;
   the coordinator rebases onto the merged Auth 2 and deletes the shim.

> **Build order:** ship (1) and (2) as self-contained commits that pass `just lint`/`deny`/tests
> with **no** dependency on Auth 2. Stage (3) as a separate commit that the coordinator merges/
> rebases after Auth 2. Flagged again in §11.

---

## 1. Dependency selection (+ `cargo deny` status)

### 1.1 `rsasl` — the SASL state machine

- **Chosen version: `rsasl = "2.3.1"`** (latest on the 2.x line, published 2026‑04‑23; MSRV
  1.65.0 ≪ workspace 1.93).
- **License: `Apache-2.0 OR MIT`** — already in the `deny.toml` allow-list (both arms allowed).
  No new license entry needed. (Verified from crates.io metadata; the v1 GPL line does **not**
  apply — v2+ was relicensed dual permissive.)
- **Features:** disable defaults and enable only what we serve, to keep the tree (and the deny
  surface) minimal and avoid pulling `libgssapi` (a native C dep) and the SSO mechanisms:

  ```toml
  rsasl = { version = "2.3.1", default-features = false, features = [
      "provider",      # the SASLServer/Session "provider" API (no base64 framing — we carry raw bytes)
      "config_builder",# SASLConfig::builder()
      "scram-sha-2",   # SCRAM-SHA-256 (+ -512, unused but cheap)
      "plain",         # PLAIN (pulls stringprep — SASLprep, wanted)
      "external",      # EXTERNAL
  ] }
  ```

  Notes:
  - `provider` (not `provider_base64`): the L0 mux envelope already carries opaque `Vec<u8>` in
    `AuthStart.initial` / `AuthStep.data` / `AuthChallenge.data`, so we step with raw bytes
    (`Session::step`, not `step64`). No double base64.
  - `scram-sha-2` pulls `hmac`, `digest`, `sha2`, `base64`, `rand`, `pbkdf2`, `stringprep`. All
    permissive (MIT/Apache/BSD/ISC/Unicode-3.0) — all already in the deny allow-list.
  - Dropping `default-features` removes `gssapi` (→ `libgssapi`/`bitflags` native link),
    `oauthbearer`/`xoauth2` (→ extra `serde_json`), `anonymous`, `login`. Smaller tree, no native
    deps, fewer advisories to track.
- **`cargo deny` expectation:** PASS. rsasl + its RustCrypto transitive deps are all permissive and
  maintained. **Action at implementation time:** run `just deny` after adding and record the result
  here; if any transitive crate trips `multiple-versions` (warn, not deny) note it but it will not
  fail the gate. No `[advisories].ignore` entry is anticipated.

### 1.2 TLS — `tokio-rustls`

- **Chosen: `tokio-rustls = "0.26"`** (current 0.26.x), which re-exports `rustls = "0.23"`. The
  workspace already pulls `rustls` transitively (genai/reqwest/hf-hub/matrix all use rustls TLS), so
  this aligns versions rather than adding a new TLS stack. **Pin to the same `rustls` 0.23 minor**
  the rest of the tree resolves (verify with `cargo tree -i rustls` at implementation time to avoid
  a `multiple-versions` warning).
- **`rustls-pemfile = "2"`** — parse PEM cert/key files from the `[api].tls_cert`/`tls_key` paths.
- **License:** rustls and tokio-rustls are `Apache-2.0 OR ISC OR MIT`; rustls-pemfile `Apache-2.0 OR
  ISC OR MIT`. All in the allow-list. PASS expected.
- **Crypto provider:** rustls 0.23 needs an explicit `CryptoProvider`. Use `aws-lc-rs` only if it is
  already the resolved default in the tree; otherwise prefer `ring` to avoid a second crypto backend
  (another `multiple-versions` risk + a heavier build). **Verify with `cargo tree` which provider
  the existing rustls users pull and match it.** Install it process-wide once
  (`rustls::crypto::<provider>::default_provider().install_default()`), or build the
  `ServerConfig` with an explicit provider.
- These deps go on **`daemon-host`** only (the transport crate). `daemon-auth` stays
  transport/TLS-free.

### 1.3 SCRAM derivation helpers (in `daemon-auth`)

To derive/persist SCRAM material on password set we need PBKDF2-HMAC-SHA256 + HMAC-SHA256 + SHA256.
`daemon-auth` already has `sha2`. Add:

```toml
pbkdf2 = { version = "0.12", default-features = false }   # PBKDF2 core (no auto-hmac wiring)
hmac   = "0.12"
# sha2 already present
```

All RustCrypto, permissive, already represented in the tree. (Alternative: reuse rsasl's internal
SCRAM helpers — but rsasl 2.x does not export a stable "derive stored password" helper, so deriving
ourselves with the RustCrypto primitives is the predictable path and keeps `daemon-auth` independent
of rsasl. **Decision: derive in `daemon-auth` with RRustCrypto primitives; do not depend on rsasl
from `daemon-auth`.**)

---

## 2. Module / file layout

```
crates/substrate/daemon-auth/
  Cargo.toml                      # + pbkdf2, hmac
  src/scram.rs        (NEW)       # RFC5802 derivation: ScramMaterial {salt, iterations, stored_key, server_key}
  src/store.rs        (EDIT)      # set_scram_credentials / scram_credentials_for; derive on create_user/set_password
  src/lib.rs          (EDIT)      # pub mod scram; re-exports

crates/substrate/daemon-host/
  Cargo.toml                      # + rsasl, tokio-rustls, rustls-pemfile, daemon-auth (workspace dep)
  src/authn.rs        (NEW)       # Authenticator: rsasl SessionCallback over AuthStore; AuthExchange state machine
  src/tls.rs          (NEW)       # serve_api_tls_tcp + ServerConfig builder from [api] config
  src/socket.rs       (EDIT)      # serve_mux auth state machine; shared frame helpers reused for TCP
  src/config.rs       (EDIT?)     # ApiTransportConfig (or keep in bins/daemon/config.rs — see §7)
  src/lib.rs          (EDIT)      # pub mod authn/tls; re-exports

bins/daemon/
  src/config.rs       (EDIT)      # [api] tls_addr/tls_cert/tls_key/require_client_cert/local_trust
  src/main.rs         (EDIT)      # build AuthStore + Authenticator + TLS; bind unix (plaintext) + tcp (tls)

Cargo.toml (workspace)            # + daemon-auth path dep; + rsasl/tokio-rustls/rustls-pemfile/pbkdf2/hmac to [workspace.dependencies]
```

`daemon-auth` is already a workspace **member** (`crates/*/*`) but is **not yet** in
`[workspace.dependencies]` and is depended on by nothing — adding the path entry + the `daemon-host`
dependency is part of deliverable (1)/(3).

---

## 3. SCRAM material derivation (RFC 5802) — deliverable (1)

The `scram_credentials` table already exists with exactly the right columns:
`(user_id, mechanism, salt BLOB, iterations INTEGER, stored_key BLOB, server_key BLOB)`. We populate
the `SCRAM-SHA-256` row whenever a password is set.

### 3.1 Derivation (`scram.rs`)

```
iterations  = SCRAM_DEFAULT_ITERATIONS   // 4096 is the RFC floor; choose >= 4096. Proposal: 4096
                                         // for interop parity with the RFC 5802 §5 test vector test,
                                         // configurable constant. (Argon2id remains the real
                                         // password-at-rest hash; SCRAM iterations are a wire KDF.)
salt        = 16 random bytes (getrandom)
SaltedPassword = PBKDF2-HMAC-SHA256(normalize(password), salt, iterations, dkLen=32)
ClientKey   = HMAC-SHA256(SaltedPassword, "Client Key")
StoredKey   = SHA256(ClientKey)
ServerKey   = HMAC-SHA256(SaltedPassword, "Server Key")
```

- `normalize(password)`: apply SASLprep (RFC 4013) via `stringprep::saslprep`. To avoid pulling
  `stringprep` into `daemon-auth` twice, either (a) add `stringprep` to `daemon-auth`, or (b) accept
  that the password was already SASLprep'd by rsasl on the wire and store the prepped form. **Decision:
  store the SASLprep-normalized password's derivation** so the persisted StoredKey/ServerKey match
  what rsasl computes from the client's prepped input. Add `stringprep` to `daemon-auth` (small,
  permissive). Note the edge case in tests (non-ASCII passwords).
- Only `StoredKey`/`ServerKey`/`salt`/`iterations` are persisted — never `SaltedPassword`,
  `ClientKey`, or the password. This is the SCRAM property of "server compromise ≠ password".

### 3.2 Store API (`store.rs`)

```rust
pub struct ScramMaterial { pub salt: Vec<u8>, pub iterations: u32,
                           pub stored_key: Vec<u8>, pub server_key: Vec<u8> }

impl AuthStore {
    /// Derive + upsert the SCRAM-SHA-256 row for a user (called by create_user + set_password).
    pub fn set_scram_credentials(&self, user_id: &str, password: &str) -> Result<()>;
    /// Fetch the persisted SCRAM material for a user+mechanism (None if absent).
    pub fn scram_credentials_for(&self, user_id: &str, mechanism: &str)
        -> Result<Option<ScramMaterial>>;
}
```

- **`create_user` and `set_password` both also derive and upsert** the SCRAM row in the *same*
  connection/transaction as the Argon2 PHC write, so PLAIN (Argon2) and SCRAM stay coherent for a
  user. (Argon2id PHC stays the source of truth for PLAIN; SCRAM material is the parallel wire-KDF
  representation.)
- `set_disabled(true)` already deletes sessions; SCRAM rows are left (deriving on next set_password
  overwrites) — but a disabled user is rejected before any SCRAM step (§4.4).
- Backfill: existing users created before this lands have a `password_credentials` row but no
  `scram_credentials` row → they cannot use SCRAM until the next `set_password`. We cannot derive
  SCRAM material from the Argon2 PHC (different KDF). **Documented limitation**: SCRAM requires a
  (re)set password; PLAIN-over-TLS still works for legacy users. Note in the access-control spec.

---

## 4. The authenticator — `rsasl` `SessionCallback` over `AuthStore` (deliverable 1)

`daemon-host/src/authn.rs`. Transport-agnostic; consumes only `daemon_auth::AuthStore` + the wire
frame types. The mux/TLS loops drive it.

### 4.1 Types

```rust
pub struct Authenticator {
    store: Arc<AuthStore>,
    config: Arc<rsasl::config::SASLConfig>,   // built once, holds the AuthCallback
    mechanisms: Vec<String>,                  // advertised order, TLS-gated (see 4.5)
}

/// What a successful exchange yields to the transport.
pub struct AuthSuccess { pub principal: Principal, pub token: String }

/// Per-connection exchange state the mux loop owns.
pub struct AuthExchange { session: rsasl::prelude::Session<DaemonValidation>, /* + flags */ }
```

- A `Validation` marker `DaemonValidation` whose `Value = ValidatedIdentity { user_id, username }`
  is set by the callback's `validate()` (PLAIN/EXTERNAL) and pulled out via `session.validation()`
  after the exchange completes. For SCRAM, the authenticated username comes from the SCRAM authcid
  property; we resolve it to a user in a post-step lookup.

### 4.2 The callback (`SessionCallback`)

One `AuthCallback { store, tls_state }` implementing `rsasl::callback::SessionCallback`:

- **`callback()` — supply credentials/data the mechanism requests (`Request`):**
  - **SCRAM-SHA-256** server side requests the stored password material. rsasl exposes this via the
    SCRAM `ScramStoredPassword` property (fields: `iterations`, `salt`, `stored_key`, `server_key`)
    — i.e. exactly our `scram_credentials` columns. The callback:
    1. reads the requested authcid (username) from the `Context`,
    2. `store.find_user(username)` → if absent / disabled, **still satisfy with a decoy** (see 4.4
       anti-oracle) or let validation fail uniformly,
    3. `store.scram_credentials_for(user_id, "SCRAM-SHA-256")` → fill the `ScramStoredPassword`
       property. rsasl then performs the proof verification itself.
    > **VERIFY at implementation:** the exact rsasl 2.3.1 property name/shape for server SCRAM
    > stored material (`properties::ScramStoredPassword` vs a `tag`-based provider). The schema was
    > clearly designed around it; confirm field-for-field and adjust the `callback` body. This is
    > the single highest-risk API detail — pin it with a SCRAM round-trip test (§9) first.
  - **PLAIN** requests nothing through `callback`; it is handled in `validate()`.
  - **EXTERNAL** authorization id is provided by the transport (the verified cert fingerprint),
    injected as a `Context` property by the mux loop before stepping (see 4.3/4.5).

- **`validate()` — finalize PLAIN / EXTERNAL and emit the identity:**
  - **PLAIN:** read authcid + password from the `Context`. Enforce **TLS-only** (4.5): if the
    connection is not TLS, return a uniform validation failure. Verify via
    `store.authenticate_password(user, pass)` (the existing Argon2id path). On success set
    `validate` value = `ValidatedIdentity`.
  - **EXTERNAL:** the transport has put the verified client-cert fingerprint into the context. Map
    fingerprint → user (see 4.6). Unmapped ⇒ validation failure (deny).
  - **SCRAM:** rsasl validates the proof internally; `validate()` resolves the authenticated authcid
    to a `ValidatedIdentity`.

### 4.3 Driving an exchange (the function the transport calls)

```rust
impl Authenticator {
    /// Begin: pick mechanism (must be advertised + TLS-permitted), feed `initial`.
    pub fn start(&self, mechanism: &str, initial: &[u8], tls: TlsState)
        -> Result<AuthStep, AuthReject>;
    /// Continue a multi-step mechanism with the client's AuthStep bytes.
    pub fn step(&mut self, exchange: &mut AuthExchange, data: &[u8])
        -> Result<AuthStep, AuthReject>;
}

pub enum AuthStep {
    Challenge(Vec<u8>),         // -> WireS2C::AuthChallenge
    Done(AuthSuccess),          // resolve principal, mint token -> WireS2C::AuthOk
}
pub enum AuthReject { /* uniform, coarse */ }   // -> WireS2C::AuthError { reason }
```

- On `AuthStep::Done`, the authenticator resolves `Principal` via
  `store.principal_for_user(user_id, username)` and mints a token via
  `store.mint_session(user_id, DEFAULT_SESSION_TTL_SECS, method)` where `method` ∈
  {`"scram-sha-256"`,`"plain"`,`"external"`}. **Token is minted only here** — never on a challenge
  or a failure.
- **AuthResume** is handled directly by the transport (not rsasl): `store.principal_for_token(token)`
  → on Ok, bind principal + (optionally) re-mint/refresh a token → `AuthOk`; on Err(NotFound/
  Disabled) → `AuthError`. (No mechanism state machine needed for resume.)

### 4.4 Anti-oracle (wrong password == unknown user)

- `AuthStore::authenticate_password` already returns the opaque `InvalidCredentials` for both
  unknown-user and bad-password. PLAIN maps any failure to one `AuthError { reason: "authentication
  failed" }`.
- SCRAM: an unknown user must not be distinguishable from a wrong password by timing or by protocol
  divergence. Standard mitigation: when the user/material is absent, serve SCRAM with a **deterministic
  decoy** salt+iterations derived from HMAC(server_secret, username) and let the proof fail at the
  final step (so the server still emits a server-first message and the failure happens at the same
  step as a real bad password). **Decision:** implement the decoy path; the `unknown_user ==
  bad_password` test (§9) asserts identical `AuthError` and identical step count. If rsasl 2.3.1
  cannot inject a decoy cleanly, fall back to a uniform early `AuthError` for both unknown-user and
  bad-proof (still no oracle, slightly weaker timing parity) — document whichever is chosen.

### 4.5 Mechanism policy + TLS gating

- Advertised set computed per-transport at `Hello` time:
  - **Unix socket (plaintext):** advertise `["SCRAM-SHA-256"]` (SCRAM is safe without TLS since the
    password never crosses). **Not** `PLAIN` (would send the password in cleartext), **not**
    `EXTERNAL` (no client cert on a Unix socket). If `[api].local_trust` is set, the Unix socket may
    skip auth entirely (see §7/§8) — but when auth is enforced, only SCRAM.
  - **TLS TCP:** advertise `["SCRAM-SHA-256", "EXTERNAL", "PLAIN"]` (SCRAM preferred first per spec;
    EXTERNAL when a client cert was presented; PLAIN allowed because the channel is encrypted).
- `TlsState { is_tls: bool, peer_cert_fingerprint: Option<String> }` is passed into `start`/the
  callback context. PLAIN with `is_tls == false` ⇒ reject. EXTERNAL with no `peer_cert_fingerprint`
  ⇒ reject.

### 4.6 EXTERNAL cert → user mapping

- The verified client-cert SHA-256 fingerprint (hex) maps to a user. **Storage decision:** reuse the
  existing `api_keys` table is wrong (that's token hashes); instead map the fingerprint to a username
  via a small convention: the cert's CN/SAN is treated as the username and looked up
  (`store.find_user(cn)`), **and** the fingerprint must match a value the admin pinned. For Auth 3
  scope (no admin API yet), the minimal, testable rule:
  - Map `fingerprint → user` through a new `daemon-auth` helper backed by a column/table. To avoid a
    migration in this track, **Decision:** add an `external_identities (user_id, fingerprint)` table
    via an additive `M::up` migration step (the migration ladder supports appending), OR (simpler)
    treat the cert CN as the username and require the user to exist + be enabled, with the
    fingerprint recorded in the audit log. **Pick the table approach** for a real fingerprint pin;
    coordinate the migration with Track C (which also adds columns) so the schema-golden refresh
    happens once. Flag to coordinator: this is a small additive migration — confirm no collision
    with Auth 4's ownership migration.
  - Unmapped fingerprint ⇒ deny (tested).

---

## 5. Session tokens & `AuthResume`

- Mint: only on success, `store.mint_session(user_id, DEFAULT_SESSION_TTL_SECS, method)` → plaintext
  token returned in `WireS2C::AuthOk { token, principal: PrincipalView::from(principal) }`.
- `PrincipalView` construction: map `Principal { user_id, username, roles, capabilities }` →
  `PrincipalView { user_id, username, roles: roles.iter().map(Role::as_str), capabilities:
  caps.iter().map(Capability snake_case) }`. Add a `From<&Principal> for PrincipalView` helper
  (likely in `daemon-host` to avoid a `daemon-api → daemon-auth` dep; confirm dependency direction —
  `daemon-api` is a contract crate and should **not** depend on `daemon-auth`, so the mapping lives
  in `daemon-host`).
- Resume: `WireC2S::AuthResume { token }` → `store.principal_for_token(token)`:
  - Ok(principal) ⇒ bind principal, optionally refresh token (re-mint + revoke old, or just reuse) →
    `AuthOk`. **Decision:** reuse the presented token (no re-mint) to keep resume idempotent; the
    `AuthOk.token` echoes the same token. (Re-mint+rotate is a later hardening.)
  - Err(NotFound) [unknown/expired/revoked] or Err(Disabled) ⇒ `AuthError { reason: "session invalid" }`.

---

## 6. TLS listener — `serve_api_tls_tcp` (deliverable 2)

`daemon-host/src/tls.rs` + `socket.rs` refactor.

### 6.1 Frame helper reuse

`socket.rs`'s `read_frame`/`write_frame` are generic over `AsyncRead/AsyncWrite + Unpin`, and
`serve_mux` takes `OwnedReadHalf`/`OwnedWriteHalf` (Unix-specific). **Refactor:** generalize
`serve_mux`/`serve_conn`/`serve_legacy` to be generic over the split halves (`R: AsyncRead`,
`W: AsyncWrite`) so the same loop serves a `tokio::net::TcpStream` wrapped by
`tokio_rustls::server::TlsStream`. `tokio_rustls::TlsStream::into_split` (via `tokio::io::split`)
yields the needed halves. This refactor is mechanical and is part of deliverable (2)/(3).

### 6.2 `serve_api_tls_tcp`

```rust
pub async fn serve_api_tls_tcp(
    listener: TcpListener,
    tls: Arc<rustls::ServerConfig>,
    api: Arc<dyn NodeApi>,
    auth: Arc<Authenticator>,        // (3) — TCP ALWAYS requires auth
) { /* accept loop: TlsAcceptor::from(tls).accept(stream) -> serve_conn_tls */ }
```

- Per accept: `acceptor.accept(tcp).await` → on handshake error, log + drop (no panic). On success,
  extract the peer cert (if any) from the `ServerConnection` (`peer_certificates()`), compute the
  SHA-256 fingerprint, build `TlsState { is_tls: true, peer_cert_fingerprint }`, then run the same
  mux loop with auth **required** (no legacy one-shot on TCP — TCP is mux+auth only).
- **TCP always requires auth**: the TCP mux loop sets `auth_required = true` unconditionally; a
  `Call`/`Open` before `AuthOk` ⇒ `Reply/End` with `ApiError::Unauthenticated` and the connection
  stays unelevated.

### 6.3 `ServerConfig` builder + cert story

`tls.rs::build_server_config(cfg: &ApiTlsConfig) -> anyhow::Result<Arc<ServerConfig>>`:

- Load cert chain from `tls_cert` (PEM, `rustls_pemfile::certs`) and private key from `tls_key`
  (`rustls_pemfile::private_key`).
- `require_client_cert`:
  - `false` ⇒ `ServerConfig::builder().with_no_client_auth()`. Clients authenticate via SCRAM/PLAIN
    over the server-authenticated TLS channel. A client cert, if presented, is still captured for
    optional EXTERNAL.
  - `true` ⇒ build a `WebPkiClientVerifier` (from `rustls::server::WebPkiClientVerifier`) over a
    configured client-CA roots bundle (`tls_client_ca` — add to config) so **untrusted client certs
    are rejected at the TLS layer** (mTLS). Then EXTERNAL maps the verified fingerprint → user.
- Crypto provider installed once at startup (see §1.2).
- **Cert story for tests:** generate ephemeral self-signed server + client certs at test time with
  `rcgen` (dev-dependency) — small, permissive (MPL-2.0/Apache, in allow-list). Verify `rcgen` is
  acceptable to deny; if not, ship static PEM fixtures under `tests/fixtures/`.

---

## 7. Config — `[api]` table

Add to `bins/daemon/src/config.rs` (the composition-layer config; env wins over TOML), and a small
resolved `ApiConfig` struct:

```toml
[api]
tls_addr            = "0.0.0.0:8443"   # absent => no TLS TCP listener (Unix-only, as today)
tls_cert            = "/path/server.pem"
tls_key             = "/path/server.key"
require_client_cert = false            # true => mTLS; EXTERNAL enabled
tls_client_ca       = "/path/ca.pem"   # required when require_client_cert = true
local_trust         = "system"         # optional: username the local Unix socket runs as (skips auth)
auth_db             = "<data_dir>/auth.sqlite"  # AuthStore path (default under data_dir)
```

Env mirrors: `DAEMON_API_TLS_ADDR`, `DAEMON_API_TLS_CERT`, `DAEMON_API_TLS_KEY`,
`DAEMON_API_REQUIRE_CLIENT_CERT`, `DAEMON_API_TLS_CLIENT_CA`, `DAEMON_API_LOCAL_TRUST`,
`DAEMON_API_AUTH_DB`.

- `tls_addr = None` ⇒ no TCP listener; the node serves Unix-only exactly as today (back-compat).
- `local_trust`:
  - `Some(username)` ⇒ the **Unix socket** binds that user's `Principal` (via
    `store.principal_for_user`/`find_user`) into the request context **without** an auth exchange,
    and advertises no mechanisms on the Unix `Hello`. This is the explicit, tested local-trust seam
    from the spec (FFI/CLI keep working).
  - `None` ⇒ the Unix socket **requires** SCRAM auth too (advertise `["SCRAM-SHA-256"]`). Default
    for a fresh secured node; but to preserve current behavior the **migration default is
    `local_trust = "system"`** with a seeded `system` admin user, so existing launches keep working.
    Coordinator decides the default polarity (it interacts with Track A's fail-closed inversion).

`[api]` parsing lives in `bins/daemon/config.rs` (like every other table) and resolves into an
`ApiConfig` passed to `run_as_host`. The TLS `ServerConfig` building lives in `daemon-host/tls.rs`.

---

## 8. `serve_mux` integration — the convergence step (deliverable 3)

### 8.1 The state machine

Per mux connection, track:

```rust
enum ConnAuth {
    Unauthenticated,                 // no Call/Open allowed (except on local_trust Unix)
    InProgress(AuthExchange),        // mid SASL exchange
    Authenticated(Principal),        // bound; Call/Open dispatch under request context
}
let auth_required: bool;             // true on TCP; true on Unix unless local_trust
let tls_state: TlsState;
```

Frame handling (replacing the current `AuthStart|AuthStep|AuthResume => AuthError` arm and gating
`Call`/`Open`):

- **`Hello`** → reply with features + `auth_mechanisms` = (auth_required ? authenticator.advertised(
  tls_state) : `[]`). When `local_trust` and Unix, bind the local principal immediately
  (`Authenticated(local_principal)`), advertise `[]`.
- **`AuthStart { mechanism, initial }`**:
  - validate mechanism is advertised + TLS-permitted; else `AuthError` (connection stays
    `Unauthenticated`, unelevated).
  - `authenticator.start(mechanism, initial, tls_state)`:
    - `Challenge(b)` → `AuthChallenge { data: b }`, state `InProgress`.
    - `Done(success)` → bind `Authenticated(principal)`, `AuthOk { token, principal_view }`.
    - `Err` → `AuthError`, state unchanged (`Unauthenticated`).
- **`AuthStep { data }`** (only valid when `InProgress`; else `AuthError`): `authenticator.step` →
  `AuthChallenge` / `AuthOk` / `AuthError` as above.
- **`AuthResume { token }`** → `store.principal_for_token` → `AuthOk` (bind) / `AuthError`.
- **`Call`/`Open`** when **not** `Authenticated` and `auth_required`:
  - `Call` ⇒ `Reply { id, res: ApiResponse::Error(ApiError::Unauthenticated("authenticate first")) }`.
  - `Open` ⇒ `End { id, error: Some(ApiError::Unauthenticated(..)) }`.
  - **connection stays unelevated** (no state change; subsequent frames still gated).
- **`Call`/`Open`** when `Authenticated(principal)` (or not `auth_required`):
  - run dispatch under the principal's request context + the authorize gate (8.2).

Per-`Call` task currently `tokio::spawn`s `dispatch(api, req)`. The authenticated principal is
`Clone` (it is `Principal: Clone`), so each spawned task captures a clone and establishes the
task-local context itself.

### 8.2 Assumed Track-A interfaces (Auth 2 — NOT yet merged)

We build deliverable (3) against these **exact assumed signatures**. A thin local shim
(`daemon-host/src/authn_ctx_shim.rs`, behind `#[cfg(...)]` or just a private module the coordinator
deletes) provides no-op/identity versions so this worktree compiles and tests independently.

```rust
// Expected in daemon-host (Track A, request-context module), task-local + default-deny.
pub async fn with_request_context<F, T>(ctx: RequestContext, fut: F) -> T
where F: Future<Output = T>;

pub struct RequestContext {
    pub principal: Principal,           // daemon_auth::Principal
    pub origin: Option<Origin>,         // existing daemon_protocol::Origin, if relayed
}

/// Read the current task's principal (None outside a context == default-deny).
pub fn current_principal() -> Option<Principal>;

/// The capability gate: map a request to its required capability and check the current principal.
/// Returns Err(ApiError::Forbidden) / Err(ApiError::Unauthenticated) when denied.
pub fn authorize(req: &ApiRequest) -> Result<(), ApiError>;
// (Track A owns required_capability(&ApiRequest) -> Capability with an exhaustive match / no `_` arm.)
```

Integration call shape inside the per-`Call` task:

```rust
let ctx = RequestContext { principal, origin: None };
let res = with_request_context(ctx, async {
    match authorize(&req) {
        Ok(()) => dispatch(api.as_ref(), req).await,
        Err(e) => ApiResponse::Error(e),
    }
}).await;
```

**Assumptions to confirm with the coordinator / Auth 2 author:**
1. `with_request_context` is **async** and wraps a future (task-local set for the duration), vs a
   guard object (`enter()` returning a drop-guard). If it's a guard, the integration wraps the
   spawned task body instead. Build the shim to match whatever Auth 2 ships; flag the shape.
2. `authorize` takes `&ApiRequest` and reads the principal from the task-local (so it must be called
   **inside** `with_request_context`). Alternative: `authorize(principal, req)` explicit — supportable
   either way; assumed implicit form.
3. The gate distinguishes `Unauthenticated` (no principal) from `Forbidden` (principal lacks cap).
   For pre-auth `Call`/`Open`, Track B emits `Unauthenticated` directly without entering a context.
4. `Principal` is the same `daemon_auth::Principal` (it is) — no separate Track-A identity type.
5. Track A inverts `commands.rs:213` polarity + introduces the local-trust principal; Track B's
   `local_trust` config feeds that principal in for the Unix path. Confirm who owns constructing the
   local-trust `Principal` (proposal: Track B builds it from `AuthStore`, Track A consumes it).

### 8.3 Streaming + cancel

The existing `streams` map + `Cancel` handling is unchanged; streams are only spawnable once
`Authenticated` (the `Open` gate in 8.1 covers it).

---

## 9. Test plan (the safety guards the plan MUST encode)

Unit tests live with their crate; integration/transport tests as `daemon-host` integration tests
(`crates/substrate/daemon-host/tests/…`) and/or extend `tests/daemon-conformance`.

### 9.1 `daemon-auth` (deliverable 1, no transport)
- **SCRAM derivation matches RFC 5802 §5 test vector** — feed the RFC's `user=user`,
  `password=pencil`, `salt`, `i=4096` and assert StoredKey/ServerKey equal the published values
  (the canonical correctness anchor).
- `set_scram_credentials` then `scram_credentials_for` round-trips; `create_user`/`set_password`
  both populate the SCRAM row; no SaltedPassword/ClientKey/password persisted.
- Non-ASCII password is SASLprep-normalized consistently (derive twice == equal).

### 9.2 Authenticator (deliverable 1)
- **SCRAM-SHA-256 full round-trip** against a test client (rsasl `SASLClient` in the test, or RFC
  vectors driving the raw bytes): correct password ⇒ `AuthOk` + principal with the user's caps + a
  minted token that `principal_for_token` resolves.
- **Wrong password == unknown user**: both yield the *same* `AuthError` and the *same* number of
  steps (no probing oracle). Asserted for SCRAM and PLAIN.
- **Token minted only on success**: a failed/aborted exchange leaves `auth_sessions` empty (count
  rows before/after).
- **Expired / revoked / disabled-user tokens rejected** on `AuthResume`: `principal_for_token`
  returns NotFound/Disabled ⇒ `AuthError`. (Reuses store guarantees; assert at the authenticator
  boundary.)
- **PLAIN over non-TLS is refused**; PLAIN over TLS verifies via Argon2id.
- **EXTERNAL**: mapped fingerprint ⇒ principal; **unmapped fingerprint denies**; missing client cert
  on EXTERNAL denies.
- **Malformed / unknown-mechanism `AuthStart`** ⇒ `AuthError` (not a panic, not a protocol kill).

### 9.3 TLS listener (deliverable 2)
- TLS handshake succeeds with the server cert; a plaintext TCP client gets no service (handshake
  fails cleanly).
- `require_client_cert = true`: a client with an **untrusted** cert is rejected at the TLS layer
  (handshake fails / connection dropped); a client with a CA-signed cert handshakes and its
  fingerprint is captured.
- `require_client_cert = false`: handshake with no client cert succeeds (SCRAM/PLAIN still required).
- Unix socket stays plaintext and (with `local_trust`) serves without auth; (without `local_trust`)
  advertises SCRAM and requires it.

### 9.4 serve_mux integration (deliverable 3 — convergence)
- **Pre-auth `Call` ⇒ `ApiError::Unauthenticated`** and the connection stays unelevated (a
  subsequent `Call` before `AuthOk` is *also* `Unauthenticated`); after `AuthOk` the same `Call`
  succeeds.
- **Pre-auth `Open` ⇒ `End { error: Unauthenticated }`**, stream not started.
- `Hello` advertises mechanisms (TLS: SCRAM,EXTERNAL,PLAIN; Unix: SCRAM or none under local_trust).
- After `AuthOk`, dispatch runs **under the principal's request context** and the authorize gate is
  consulted (assert via a stubbed `authorize`/`current_principal` shim: a Viewer principal is
  `Forbidden` on a write op; an Operator is allowed) — wired fully once Auth 2 merges.
- **TCP always requires auth**; **Unix under local_trust binds the local principal** (an op runs as
  `system`); the negative is also tested (no local_trust ⇒ Unix requires SCRAM).
- `AuthResume` happy-path reconnect binds the principal without a mechanism exchange.

### 9.5 deny / lint
- `just deny` clean (record rsasl/tokio-rustls/rcgen results here at impl time).
- `just lint` clean (rustfmt + clippy -D warnings + spell + secrets — no cert/key material or tokens
  in tracked files; test PEMs are clearly fixtures or generated).

---

## 10. Step-by-step implementation sequence

1. **(1a)** `daemon-auth`: add `scram.rs` (+ deps `pbkdf2`,`hmac`,`stringprep`), `set_scram_credentials`/
   `scram_credentials_for`, derive on create/set_password. Tests 9.1. → commit, `just lint`/`deny`/test.
2. **(1b)** `daemon-host`: add `daemon-auth` dep + `authn.rs` (rsasl callback + exchange). Tests 9.2.
   → commit (independent of Auth 2).
3. **(2)** `daemon-host`: generalize frame loop over generic IO; add `tls.rs` + `serve_api_tls_tcp`;
   `bins/daemon` `[api]` config + bind the TLS listener (auth required) and keep Unix plaintext.
   Tests 9.3. → commit (independent of Auth 2).
4. **(3)** `socket.rs` serve_mux auth state machine + Track-A shim + `with_request_context`/`authorize`
   integration; `main.rs` builds the `AuthStore` + `Authenticator` + local-trust principal and wires
   both transports. Tests 9.4. → **staged commit for the coordinator to rebase onto merged Auth 2.**

---

## 11. Summary, Track-A assumptions, blockers

### Summary
- `rsasl 2.3.1` (Apache-2.0 OR MIT — in the deny allow-list), `default-features = false` + features
  `provider, config_builder, scram-sha-2, plain, external`. `tokio-rustls 0.26` (re-exports the
  tree's existing `rustls 0.23`) + `rustls-pemfile 2`. SCRAM material derived in `daemon-auth` with
  RustCrypto `pbkdf2`/`hmac`/`sha2` and persisted into the existing `scram_credentials` columns; the
  authenticator serves SCRAM/PLAIN(TLS-only)/EXTERNAL(mTLS) via one `SessionCallback` over
  `AuthStore`, mints an opaque session token only on success, and supports `AuthResume`.
- Deliverables (1) rsasl-authenticator and (2) tls-listener land **independently first** (no Auth 2
  dependency); (3) serve_mux integration is the **convergence commit** the coordinator sequences
  after Auth 2 merges, built here against an assumed-interface shim.

### Track-A interface assumptions (to validate before convergence)
1. `with_request_context(ctx, fut).await` — async future-wrapping task-local (vs an `enter()` guard).
2. `authorize(&ApiRequest) -> Result<(), ApiError>` reading the task-local principal (vs explicit
   `authorize(&principal, &req)`).
3. The gate separates `Unauthenticated` (no principal) from `Forbidden` (missing capability); Track B
   emits `Unauthenticated` for pre-auth `Call`/`Open` without entering a context.
4. `RequestContext { principal: daemon_auth::Principal, origin: Option<Origin> }`; `Principal` is the
   shared `daemon_auth` type (confirmed).
5. Ownership of the **local-trust principal**: proposal — Track B constructs it from `AuthStore`
   (`[api].local_trust` username), Track A consumes it via the request context; Track A owns the
   `commands.rs:213` polarity inversion.

## 12. Implementation results — deliverables (1) and (2) [LANDED]

Deliverables (1) rsasl-authenticator and (2) tls-listener are implemented, gated, and committed.
Deliverable (3) + the two held cross-track decisions remain for the coordinator.

### Pinned dependency facts (B1 / B4 verified)
- **rsasl `2.3.1`**, `default-features = false` + `["provider","config_builder","scram-sha-2","plain",
  "external"]`. Server-side SCRAM uses the **`ScramStoredPassword { iterations, salt, stored_key,
  server_key }`** property (its fields map 1:1 onto the `scram_credentials` columns) supplied from
  `AuthStore` in `SessionCallback::callback`. **Critical pin:** the rsasl SCRAM server returns
  `Ok(State::Finished)` even on a *bad* proof (writing a `ServerFinal::Error`) and invokes
  `SessionCallback::validate` **only after the proof verifies** — so the authenticator captures the
  identity (and mints the token) in `validate`, never in `callback`. The RFC 7677 §3 vector test +
  a live `rsasl` `SASLClient` round-trip both pass.
- **TLS:** `tokio-rustls 0.26` over the tree's already-resolved **`rustls 0.23.41`**; crypto provider
  pinned to **aws-lc-rs** (`cargo tree -i rustls` → rustls 0.23 + aws-lc-rs active via reqwest's
  `__rustls-aws-lc-rs`; both ring + aws-lc-rs compile in the tree, aws-lc-rs is the rustls default).
  No second crypto backend introduced. PEM loading uses the maintained **`rustls-pki-types`
  `PemObject`** reader (NOT `rustls-pemfile`, which is now unmaintained — see below).
- SCRAM derivation in `daemon-auth` uses RustCrypto `pbkdf2`/`hmac`/`sha2` + `stringprep` (SASLprep);
  `daemon-host` adds `hmac`/`getrandom` for the deterministic anti-oracle decoy.

### `cargo deny` result (recorded per directive item 4)
- `cargo deny check advisories licenses sources` → **advisories ok, licenses ok, sources ok.** All
  new deps (rsasl, tokio-rustls, rcgen[dev], pbkdf2, hmac, stringprep) are permissive
  (Apache-2.0/MIT/ISC/BSD) and already allow-listed; no new `[advisories].ignore` entry.
- **Mid-flight finding:** the initial plan's `rustls-pemfile` dep tripped **RUSTSEC-2025-0134**
  (rustls-pemfile unmaintained / archived Aug 2025). Resolved by dropping it entirely and parsing
  PEM via the `PemObject` trait already provided by `rustls-pki-types` (in-tree via rustls 0.23) —
  fewer deps, advisory cleared.
- `cargo deny check bans` → **FAILS pre-existingly** with `error[wildcard]` for *every* workspace
  crate ("allow-wildcard-paths … does not apply to public crates"). Verified on a clean `HEAD`
  (`git stash`): **43 wildcard errors with none of the Auth 3 changes applied**. This is a
  repo-baseline cargo-deny-version interaction (the internal path crates lack `publish = false`),
  not introduced by Auth 3, and affects untouched crates (`daemon-cli`, `daemon-orchestration`, …)
  identically. Flagging for the coordinator; out of Track B scope to fix.

### Gates (all green for the touched crates)
- `cargo fmt --check`: clean. `cargo clippy -p daemon-auth -p daemon-host -p daemon --all-targets
  -- -D warnings`: clean. `cargo build -p daemon`: ok.
- Tests: `daemon-auth` 17 passed; `daemon-host` 81 passed (incl. 9 authenticator + 3 TLS).
  Coverage of the directive's safety guards: RFC 7677 SCRAM vector; live SCRAM round-trip;
  wrong-password == unknown-user (no oracle); token minted only on success; expired/revoked/
  disabled-user resume rejected; PLAIN refused without TLS (and verified over TLS); EXTERNAL
  unmapped/no-cert denies; unknown-mechanism rejected; TLS handshake + mTLS untrusted-client-cert
  rejected (server-authoritative); `require_client_cert` without a CA errors.

### Scope boundary actually shipped (read before deliverable 3)
- **Deliverable 1** (`crates/substrate/daemon-auth/src/scram.rs` + store methods;
  `crates/substrate/daemon-host/src/authn.rs`): SCRAM-SHA-256 + PLAIN(TLS-only) fully; EXTERNAL
  mechanism path scaffolded — fingerprint→user mapping is a stub returning `None`
  (`AuthCallback::external_identity`, marked `TODO(auth3-external)`), so EXTERNAL fail-closes until
  the `external_identities` migration (B3, held) wires the table. No migration was added.
- **Deliverable 2** (`crates/substrate/daemon-host/src/tls.rs`; `[api]` config in
  `bins/daemon/src/config.rs`; wiring in `bins/daemon/src/main.rs`): `build_server_config` +
  `serve_api_tls_tcp`. The TLS path runs an **authenticated** mux loop (TCP always requires auth;
  pre-auth `Call`/`Open` → `Unauthenticated`). It does **NOT** apply the Track-A request-context /
  authorize gate — the dispatch site carries `TODO(auth3-deliverable3)`. The existing Unix
  `serve_mux` is **untouched** (still answers `AuthError` to auth frames, as before); unifying the
  two loops + the `[api].local_trust` Unix policy (B5, held) is the convergence step.

## 13. Convergence — deliverable (3) [LANDED]

Auth 2 (`feature/auth2-authz-core` @ 90d48b2) is merged into this branch; the serve_mux
handshake → auth → context → authorize → dispatch state machine is wired into every dispatch entry
point using Auth 2's real interface (`request_context` + `authz`). No shim was used (the prior
commits left TODOs, not a shim).

- **Merge conflicts resolved:** `daemon-host/src/lib.rs` (module list + re-exports — kept BOTH
  Auth 2's `authz`/`request_context` and Auth 3's `authn`/`tls`), `daemon-host/Cargo.toml` (deduped
  the `daemon-auth` dependency both branches added), and `AUTH_PLAN.md` (add/add — kept ours). The
  workspace `Cargo.toml`/`Cargo.lock` auto-merged; `commands.rs` is Auth 2's `caller_access`
  inversion (principal-driven; kept verbatim).
- **One shared, context-aware mux loop** (`socket::serve_mux<R,W>`, `AuthMode::{LocalSystem,
  Required}`) now backs **both** the Unix socket and TLS/TCP; the per-`Call` spawned task
  re-enters `with_request_context(...)` (the task-local is not inherited by `tokio::spawn`) and runs
  `authorize` before `dispatch`. Entry points wired: Unix `serve_mux` + `serve_legacy`, TLS TCP,
  HTTP (`daemon-http`), FFI (`daemon_host_call`). Pre-auth `Call`/`Open` on an auth-required
  connection → `Unauthenticated` / `End{Unauthenticated}`, connection stays unelevated.
- **B5 ratified:** `[api].local_trust` defaults to `system` → Unix/FFI/in-process-HTTP bind
  `RequestContext::system()` (no SASL offered); disabling it makes the Unix socket require SCRAM and
  fully gates HTTP (deny-all until HTTP SASL lands). TCP/TLS always requires auth.
- **B3 ratified:** EXTERNAL fingerprint→user stays the fail-closed stub (denies); the
  `external_identities` migration is deferred to Auth 4's migration batch. TODO left in place.

### Blockers / risks (need coordinator or upstream confirmation)
- **B1 (highest):** exact rsasl 2.3.1 **server-side SCRAM stored-material property** shape
  (`ScramStoredPassword` fields vs provider tag). The schema matches the expected fields, but the
  API call must be pinned with the RFC round-trip test before building the rest. Mitigated by doing
  the SCRAM round-trip test first.
- **B2:** the SCRAM **anti-oracle decoy** path — whether rsasl lets us inject decoy salt/iterations
  for an unknown user; fallback is a uniform early `AuthError`. Affects the exact assertion in the
  `unknown_user == bad_password` test.
- **B3:** **EXTERNAL fingerprint→user mapping** needs a tiny additive migration
  (`external_identities`) or a CN-as-username convention. This adds a `daemon-auth` migration step —
  must be coordinated with Track C/Track D so the schema-golden refresh and `MIGRATIONS.validate()`
  happen once, not thrice.
- **B4:** rustls **crypto provider** (`ring` vs `aws-lc-rs`) and `rustls` minor must match the
  already-resolved tree to avoid a `multiple-versions` warning / second crypto backend — verify with
  `cargo tree -i rustls` before pinning.
- **B5:** **default of `[api].local_trust`** interacts with Track A's fail-closed inversion: a
  too-eager default that requires SCRAM on the Unix socket breaks the FFI/CLI/legacy one-shot path;
  too-lax leaves an implicit-admin local socket. Proposal: default `local_trust = "system"` with a
  seeded `system` user — coordinator to ratify.
- **B6:** **Auth 2 is not merged** — deliverable (3) cannot be verified end-to-end (gate behavior)
  until it is; it is built and unit-tested against the shim and handed to the coordinator for the
  final rebase.
