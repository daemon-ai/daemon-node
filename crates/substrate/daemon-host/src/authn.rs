// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The node authenticator: a transport-agnostic SASL state machine over [`daemon_auth::AuthStore`].
//!
//! This is deliverable (1) of the Auth 3 track. It wraps the `rsasl` SASL framework with a
//! [`SessionCallback`] backed by the identity store and serves three mechanisms per the access
//! control spec:
//!
//! * **`SCRAM-SHA-256`** — the preferred interactive mechanism. The server never sees the password;
//!   `rsasl` verifies the client proof against the stored `salt`/`iterations`/`StoredKey`/`ServerKey`
//!   ([`daemon_auth::ScramMaterial`]) that the callback supplies via the `ScramStoredPassword`
//!   property. Safe even without TLS (the password never crosses the wire).
//! * **`PLAIN`** — username + password verified against the Argon2id PHC
//!   ([`AuthStore::authenticate_password`]). **Permitted only over TLS** (the password is in the
//!   clear); a non-TLS connection refuses it.
//! * **`EXTERNAL`** — the verified mTLS client-certificate fingerprint maps to a user. The mechanism
//!   path is fully scaffolded here, but the fingerprint→user table is **not yet wired** (see the
//!   `TODO(auth3-external)` in [`AuthCallback::external_identity`]); until the coordinator sequences
//!   the `external_identities` migration with the other `daemon-auth`/`daemon-store` schema changes,
//!   EXTERNAL always denies (no mapping == deny, which is the fail-closed default).
//!
//! On success the authenticator mints an opaque session token ([`AuthStore::mint_session`]) — **only**
//! on success, never on a challenge or failure — and resolves the caller's [`Principal`]. Reconnects
//! present a prior token via [`Authenticator::resume`].
//!
//! Anti-oracle: an unknown user, a disabled user, and a user lacking SCRAM material are all served a
//! deterministic *decoy* SCRAM credential so the exchange fails at the same proof-verification step
//! as a wrong password — there is no protocol-observable difference between "no such user" and "bad
//! password" (mirroring [`AuthStore::authenticate_password`]'s coarse `InvalidCredentials`).

use std::sync::{Arc, Mutex};

use daemon_api::PrincipalView;
use daemon_auth::{
    scram::SCRAM_DEFAULT_ITERATIONS, AuthStore, Principal, ScramMaterial, DEFAULT_SESSION_TTL_SECS,
    SCRAM_SHA_256,
};
use hmac::{Hmac, Mac};
use rsasl::callback::{Context, Request, SessionCallback, SessionData};
use rsasl::config::SASLConfig;
use rsasl::mechanisms::scram::properties::ScramStoredPassword;
use rsasl::prelude::{Mechname, SASLServer, Session, SessionError, State};
use rsasl::property::{AuthId, Password};
use rsasl::validate::{NoValidation, Validate, ValidationError};
use sha2::Sha256;

/// The mechanism names the node may serve, in advertised preference order. PLAIN/EXTERNAL are
/// gated to TLS connections by [`Authenticator::advertised_mechanisms`].
pub const MECH_SCRAM_SHA_256: &str = SCRAM_SHA_256;
/// PLAIN mechanism name (TLS-only).
pub const MECH_PLAIN: &str = "PLAIN";
/// EXTERNAL mechanism name (mTLS client-cert).
pub const MECH_EXTERNAL: &str = "EXTERNAL";

/// Per-connection transport security facts the mechanism policy depends on.
#[derive(Clone, Debug, Default)]
pub struct TlsState {
    /// Whether the connection is carried over TLS (gates PLAIN).
    pub is_tls: bool,
    /// The verified client-certificate SHA-256 fingerprint (hex), if mTLS presented one. Drives
    /// EXTERNAL.
    pub peer_cert_fingerprint: Option<String>,
}

impl TlsState {
    /// A plaintext (non-TLS) connection — e.g. the local Unix socket. SCRAM only.
    pub fn plaintext() -> Self {
        Self {
            is_tls: false,
            peer_cert_fingerprint: None,
        }
    }
}

/// The identity resolved by a successful mechanism exchange (before role/token resolution).
#[derive(Clone, Debug)]
struct ValidatedIdentity {
    user_id: String,
    username: String,
}

/// A successful authentication outcome handed back to the transport.
#[derive(Clone, Debug)]
pub struct AuthSuccess {
    /// The authenticated, role-resolved principal to bind to the connection.
    pub principal: Principal,
    /// A freshly minted opaque session token (for `AuthOk`; the client presents it on reconnect).
    pub token: String,
    /// The principal as a wire view, for `WireS2C::AuthOk` (advisory client-side gating).
    pub principal_view: PrincipalView,
}

/// A coarse, non-revealing authentication failure (maps to `WireS2C::AuthError`).
#[derive(Clone, Debug)]
pub struct AuthReject {
    /// A short reason that never distinguishes "no such user" from "bad credential".
    pub reason: String,
}

impl AuthReject {
    fn failed() -> Self {
        Self {
            reason: "authentication failed".into(),
        }
    }

    fn unsupported_mechanism() -> Self {
        Self {
            reason: "unsupported mechanism".into(),
        }
    }
}

/// The result of *starting* a mechanism exchange ([`Authenticator::begin`] / [`Authenticator::resume`]).
pub enum BeginOutcome {
    /// More steps required: send these opaque bytes as `WireS2C::AuthChallenge` and await the next
    /// `WireC2S::AuthStep`, fed into the returned live exchange.
    Challenge {
        /// Opaque mechanism bytes for the client.
        data: Vec<u8>,
        /// The exchange to resume with [`AuthExchange::step`].
        exchange: AuthExchange,
    },
    /// Authentication completed in one step (PLAIN/EXTERNAL/resume).
    Success {
        /// Trailing mechanism bytes to deliver as a final `AuthChallenge` before `AuthOk`, if any.
        final_data: Option<Vec<u8>>,
        /// The success outcome.
        success: Box<AuthSuccess>,
    },
    /// Authentication failed; send `WireS2C::AuthError { reason }`, connection stays unelevated.
    Failed(AuthReject),
}

/// The result of *stepping* an in-progress exchange ([`AuthExchange::step`]).
pub enum StepOutcome {
    /// More steps required: send these bytes as `AuthChallenge` and await the next `AuthStep`.
    Challenge(Vec<u8>),
    /// Authentication completed. `final_data` (e.g. the SCRAM server-final message) must be sent as
    /// a trailing `AuthChallenge` *before* `AuthOk` (the frozen `AuthOk` carries no mechanism bytes).
    Success {
        /// Trailing mechanism bytes to deliver before `AuthOk`, if any.
        final_data: Option<Vec<u8>>,
        /// The success outcome.
        success: Box<AuthSuccess>,
    },
    /// Authentication failed; send `AuthError` and keep the connection unelevated.
    Failed(AuthReject),
}

/// Internal one-step outcome shared by `begin` and `step`.
enum Drive {
    Challenge(Vec<u8>),
    Success {
        final_data: Option<Vec<u8>>,
        success: Box<AuthSuccess>,
    },
    Failed(AuthReject),
}

/// A live, in-progress multi-step mechanism exchange (e.g. SCRAM between server-first and the final
/// proof). Owned by the transport between an `AuthChallenge` and the next `AuthStep`.
pub struct AuthExchange {
    session: Session<NoValidation>,
    captured: Arc<Mutex<Option<ValidatedIdentity>>>,
    store: Arc<AuthStore>,
    method: String,
}

/// The node authenticator. Cheap to clone-share via `Arc`; holds the identity store and a
/// process-local secret used to derive deterministic decoy SCRAM material for unknown users.
pub struct Authenticator {
    store: Arc<AuthStore>,
    decoy_secret: [u8; 32],
    /// The shared auth-audit sink (login success/failure + permission denials ride this same
    /// `node-auth` chain as the admin events). `None` => login/denial audit is a no-op (e.g. tests
    /// or a node assembled without journaling).
    audit: Option<Arc<crate::auth_audit::AuthAudit>>,
}

impl Authenticator {
    /// Build an authenticator over `store`, minting a fresh process-local decoy secret.
    pub fn new(store: Arc<AuthStore>) -> Self {
        let mut decoy_secret = [0u8; 32];
        // A failure here is catastrophic (no entropy); fall back to a fixed value so the node still
        // boots — the decoy only needs to be unpredictable-enough to not leak account existence.
        let _ = getrandom::getrandom(&mut decoy_secret);
        Self {
            store,
            decoy_secret,
            audit: None,
        }
    }

    /// Attach the shared auth-audit sink so login success/failure (and, via [`Self::audit`], the
    /// transport's permission denials) are recorded onto the verifiable `node-auth` journal stream.
    pub fn with_audit(mut self, audit: Arc<crate::auth_audit::AuthAudit>) -> Self {
        self.audit = Some(audit);
        self
    }

    /// The attached auth-audit sink, if any (the transport reaches it to record permission denials,
    /// which are decided in the capability gate, not the authenticator).
    pub fn audit(&self) -> Option<&Arc<crate::auth_audit::AuthAudit>> {
        self.audit.as_ref()
    }

    /// Record a successful login onto the audit chain (no-op without an attached sink). The SASL
    /// state machine ([`Self::begin`]/[`AuthExchange::step`]/[`Self::resume`]) is synchronous, so the
    /// transport invokes this async hook once it has the resolved principal + method.
    pub async fn audit_login_ok(&self, user_id: &str, method: &str) {
        if let Some(a) = &self.audit {
            a.login_ok(user_id, method).await;
        }
    }

    /// Record a failed login onto the audit chain (no-op without an attached sink). Never receives
    /// or records the supplied password; `username` is the attempted identity only when the mechanism
    /// exposes it.
    pub async fn audit_login_fail(&self, mechanism: &str, username: Option<&str>) {
        if let Some(a) = &self.audit {
            a.login_fail(mechanism, username).await;
        }
    }

    /// The mechanisms to advertise on a `Hello`, in preference order, given the transport security.
    /// SCRAM is always offered; PLAIN and EXTERNAL only over TLS (PLAIN sends the password; EXTERNAL
    /// needs a client certificate).
    pub fn advertised_mechanisms(&self, tls: &TlsState) -> Vec<String> {
        let mut mechs = vec![MECH_SCRAM_SHA_256.to_string()];
        if tls.is_tls {
            mechs.push(MECH_EXTERNAL.to_string());
            mechs.push(MECH_PLAIN.to_string());
        }
        mechs
    }

    /// Begin a mechanism exchange. Rejects mechanisms not permitted on this transport (e.g. PLAIN on
    /// a plaintext socket), then feeds the client's `initial` response into the first server step.
    pub fn begin(&self, mechanism: &str, initial: &[u8], tls: TlsState) -> BeginOutcome {
        if !self
            .advertised_mechanisms(&tls)
            .iter()
            .any(|m| m == mechanism)
        {
            return BeginOutcome::Failed(AuthReject::unsupported_mechanism());
        }
        let mechname = match Mechname::parse(mechanism.as_bytes()) {
            Ok(m) => m,
            Err(_) => return BeginOutcome::Failed(AuthReject::unsupported_mechanism()),
        };
        let captured = Arc::new(Mutex::new(None));
        let callback = AuthCallback {
            store: self.store.clone(),
            tls,
            decoy_secret: self.decoy_secret,
            captured: captured.clone(),
            mechanism: mechanism.to_string(),
        };
        let config = match SASLConfig::builder()
            .with_defaults()
            .with_callback(callback)
        {
            Ok(c) => c,
            Err(_) => return BeginOutcome::Failed(AuthReject::failed()),
        };
        let session = match SASLServer::<NoValidation>::new(config).start_suggested(mechname) {
            Ok(s) => s,
            Err(_) => return BeginOutcome::Failed(AuthReject::unsupported_mechanism()),
        };
        let mut exchange = AuthExchange {
            session,
            captured,
            store: self.store.clone(),
            method: mechanism.to_string(),
        };
        match exchange.drive(initial) {
            Drive::Challenge(data) => BeginOutcome::Challenge { data, exchange },
            Drive::Success {
                final_data,
                success,
            } => BeginOutcome::Success {
                final_data,
                success,
            },
            Drive::Failed(reject) => BeginOutcome::Failed(reject),
        }
    }

    /// Reconnect fast-path: resolve a previously issued opaque session token to its principal,
    /// re-issuing the same token. Unknown / expired / revoked / disabled all map to the same coarse
    /// failure.
    pub fn resume(&self, token: &str) -> BeginOutcome {
        match self.store.principal_for_token(token) {
            Ok(principal) => {
                let principal_view = principal_view(&principal);
                BeginOutcome::Success {
                    final_data: None,
                    success: Box::new(AuthSuccess {
                        principal,
                        token: token.to_string(),
                        principal_view,
                    }),
                }
            }
            Err(_) => BeginOutcome::Failed(AuthReject {
                reason: "session invalid".into(),
            }),
        }
    }
}

impl AuthExchange {
    /// Feed the next client `AuthStep` bytes into the mechanism.
    pub fn step(&mut self, data: &[u8]) -> StepOutcome {
        match self.drive(data) {
            Drive::Challenge(d) => StepOutcome::Challenge(d),
            Drive::Success {
                final_data,
                success,
            } => StepOutcome::Success {
                final_data,
                success,
            },
            Drive::Failed(reject) => StepOutcome::Failed(reject),
        }
    }

    /// Perform one server step over `input`. A token is minted (and the principal resolved) *only*
    /// when the mechanism reaches `State::Finished` **and** the callback captured a validated
    /// identity.
    fn drive(&mut self, input: &[u8]) -> Drive {
        let mut out = Vec::new();
        match self.session.step(Some(input), &mut out) {
            Ok(State::Running) => Drive::Challenge(out),
            Ok(State::Finished(_)) => {
                let identity = self
                    .captured
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
                match identity {
                    Some(id) => self.finish(id, out),
                    // Mechanism completed at the SASL layer but no identity was validated (wrong
                    // PLAIN password, unmapped EXTERNAL cert): deny without minting anything.
                    None => Drive::Failed(AuthReject::failed()),
                }
            }
            Err(_) => Drive::Failed(AuthReject::failed()),
        }
    }

    /// Resolve the principal + mint the session token for a validated identity.
    fn finish(&self, id: ValidatedIdentity, final_data: Vec<u8>) -> Drive {
        let principal = match self.store.principal_for_user(&id.user_id, &id.username) {
            Ok(p) => p,
            Err(_) => return Drive::Failed(AuthReject::failed()),
        };
        let token =
            match self
                .store
                .mint_session(&id.user_id, DEFAULT_SESSION_TTL_SECS, &self.method)
            {
                Ok(t) => t,
                Err(_) => return Drive::Failed(AuthReject::failed()),
            };
        let principal_view = principal_view(&principal);
        Drive::Success {
            final_data: (!final_data.is_empty()).then_some(final_data),
            success: Box::new(AuthSuccess {
                principal,
                token,
                principal_view,
            }),
        }
    }
}

/// The `rsasl` callback bridging mechanism property requests + validation to the identity store, for
/// one connection's chosen mechanism.
struct AuthCallback {
    store: Arc<AuthStore>,
    tls: TlsState,
    decoy_secret: [u8; 32],
    captured: Arc<Mutex<Option<ValidatedIdentity>>>,
    mechanism: String,
}

impl AuthCallback {
    /// Resolve the SCRAM stored material to serve for `username`: the real material when the user
    /// exists, is enabled, and has SCRAM material; otherwise deterministic *decoy* material so the
    /// exchange fails at proof verification exactly like a wrong password (no account-probing
    /// oracle). The identity itself is resolved later in `validate_scram`, gated on proof success.
    fn scram_material_for(&self, username: &str) -> ScramMaterial {
        if let Ok(Some(user)) = self.store.find_user(username) {
            if !user.disabled {
                if let Ok(Some(material)) =
                    self.store.scram_credentials_for(&user.id, SCRAM_SHA_256)
                {
                    return material;
                }
            }
        }
        self.decoy_material(username)
    }

    /// Deterministic decoy SCRAM material derived from the process secret + username. Stable across
    /// attempts (so the salt does not change between tries) but unrelated to any real password, so
    /// proof verification always fails.
    fn decoy_material(&self, username: &str) -> ScramMaterial {
        let salt = self.decoy_mac(b"salt", username)[..16].to_vec();
        let stored_key = self.decoy_mac(b"stored", username).to_vec();
        let server_key = self.decoy_mac(b"server", username).to_vec();
        ScramMaterial {
            salt,
            iterations: SCRAM_DEFAULT_ITERATIONS,
            stored_key,
            server_key,
        }
    }

    fn decoy_mac(&self, tag: &[u8], username: &str) -> [u8; 32] {
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&self.decoy_secret)
            .expect("HMAC accepts any key length");
        mac.update(tag);
        mac.update(b"\0");
        mac.update(username.as_bytes());
        mac.finalize().into_bytes().into()
    }

    /// SCRAM: rsasl invokes `validate()` **only after** the client proof verifies against the stored
    /// key, so by here the password is already proven. Resolve the proven username to its identity.
    /// (We re-check existence defensively; a decoy/unknown user can never reach this path because its
    /// decoy material fails proof verification.)
    fn validate_scram(&self, context: &Context<'_>) {
        let Some(username) = context.get_ref::<AuthId>() else {
            return;
        };
        if let Ok(Some(user)) = self.store.find_user(username) {
            if !user.disabled {
                *self.captured.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(ValidatedIdentity {
                        user_id: user.id,
                        username: user.username,
                    });
            }
        }
    }

    /// PLAIN: verify the username + password against the Argon2id PHC, but only over TLS. On success
    /// stash the identity; on any failure (non-TLS, unknown user, bad password, disabled) leave it
    /// unset so the transport denies uniformly.
    fn validate_plain(&self, context: &Context<'_>) {
        if !self.tls.is_tls {
            // PLAIN is forbidden on a plaintext transport — never verify the password.
            return;
        }
        let (Some(authid), Some(password)) =
            (context.get_ref::<AuthId>(), context.get_ref::<Password>())
        else {
            return;
        };
        let password = match std::str::from_utf8(password) {
            Ok(p) => p,
            Err(_) => return,
        };
        if let Ok(principal) = self.store.authenticate_password(authid, password) {
            *self.captured.lock().unwrap_or_else(|e| e.into_inner()) = Some(ValidatedIdentity {
                user_id: principal.user_id,
                username: principal.username,
            });
        }
    }

    /// EXTERNAL: map the verified client-certificate fingerprint to a user.
    fn validate_external(&self) {
        let Some(fingerprint) = self.tls.peer_cert_fingerprint.as_deref() else {
            return; // EXTERNAL with no presented client cert: deny.
        };
        if let Some(id) = self.external_identity(fingerprint) {
            *self.captured.lock().unwrap_or_else(|e| e.into_inner()) = Some(id);
        }
    }

    /// Map a verified client-cert fingerprint to a user identity via the `external_identities` table
    /// (Auth 4). An unmapped fingerprint (or a disabled mapped user) resolves to `None` — the
    /// fail-closed default, so an untrusted certificate never authenticates. Fingerprints are
    /// enrolled out of band (the store-level [`AuthStore::set_external_identity`] writer; the admin
    /// enrollment op is a later track).
    fn external_identity(&self, fingerprint: &str) -> Option<ValidatedIdentity> {
        self.store
            .external_identity(fingerprint)
            .ok()
            .flatten()
            .map(|(user_id, username)| ValidatedIdentity { user_id, username })
    }
}

impl SessionCallback for AuthCallback {
    fn callback(
        &self,
        _session_data: &SessionData,
        context: &Context<'_>,
        request: &mut Request<'_>,
    ) -> Result<(), SessionError> {
        // SCRAM asks the server for the stored credential of the client-supplied username. Supply
        // the real material, or a deterministic *decoy* for an unknown/disabled/no-SCRAM user so the
        // exchange fails at proof verification exactly like a wrong password (no account-probing
        // oracle: we always satisfy the property, so rsasl never emits the distinct "unknown user"
        // server-final). The identity is captured in `validate`, which rsasl invokes only once the
        // proof has verified — so a bad proof never mints anything.
        if request.is::<ScramStoredPassword>() {
            let username = context.get_ref::<AuthId>().unwrap_or("");
            let material = self.scram_material_for(username);
            let answer = ScramStoredPassword::new(
                material.iterations,
                &material.salt,
                &material.stored_key,
                &material.server_key,
            );
            request.satisfy::<ScramStoredPassword>(&answer)?;
        }
        Ok(())
    }

    fn validate(
        &self,
        _session_data: &SessionData,
        context: &Context<'_>,
        _validate: &mut Validate<'_>,
    ) -> Result<(), ValidationError> {
        // Branch on the connection's selected mechanism (SessionData exposes no mechanism accessor).
        // SCRAM identity is captured in `callback`; only PLAIN/EXTERNAL validate here. We never
        // return `Err` (which would be a fatal protocol abort) — a failed credential simply leaves
        // the identity unset, and the transport denies uniformly.
        match self.mechanism.as_str() {
            MECH_PLAIN => self.validate_plain(context),
            MECH_EXTERNAL => self.validate_external(),
            MECH_SCRAM_SHA_256 => self.validate_scram(context),
            _ => {}
        }
        Ok(())
    }
}

/// Project a [`Principal`] onto its wire [`PrincipalView`] (advisory client-side gating on `AuthOk`).
/// Capability names are the model's stable snake_case serde strings.
pub fn principal_view(p: &Principal) -> PrincipalView {
    PrincipalView {
        user_id: p.user_id.clone(),
        username: p.username.clone(),
        roles: p.roles.iter().map(|r| r.as_str().to_string()).collect(),
        capabilities: p.capabilities.iter().map(capability_wire_name).collect(),
    }
}

/// The stable snake_case wire name of a capability (its serde representation, the single source of
/// truth shared with the CDDL `PrincipalView`).
fn capability_wire_name(cap: &daemon_auth::Capability) -> String {
    serde_json::to_value(cap)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_auth::{Capability, Role};
    use rsasl::prelude::{SASLClient, State as ClientState};

    fn store_with_user(username: &str, password: &str, roles: &[Role]) -> Arc<AuthStore> {
        let store = Arc::new(AuthStore::open_in_memory().expect("store"));
        store.create_user(username, password, roles).expect("user");
        store
    }

    /// Outcome of a fully-mediated client/server SASL exchange (the test's stand-in transport).
    enum Mediated {
        Ok(Box<AuthSuccess>),
        Denied,
    }

    /// Drive a real rsasl `SASLClient` against the [`Authenticator`] for `mechanism`, shuttling
    /// opaque bytes between the two exactly as the wire transport would (`AuthStart`/`AuthStep`
    /// <-> `AuthChallenge`). Proves end-to-end that the persisted SCRAM material verifies against an
    /// independent SCRAM implementation (the rsasl client).
    fn mediate(
        auth: &Authenticator,
        mechanism: &str,
        username: &str,
        password: &str,
        tls: TlsState,
    ) -> Mediated {
        let client_config =
            SASLConfig::with_credentials(None, username.to_string(), password.to_string())
                .expect("client config");
        let mechname = Mechname::parse(mechanism.as_bytes()).expect("mechname");
        let mut client = SASLClient::new(client_config)
            .start_suggested_iter([mechname])
            .expect("client start");

        // All three mechanisms are client-first: the client produces the first message with no input.
        let mut cbuf = Vec::new();
        let mut cstate = client.step(None, &mut cbuf).expect("client first step");

        let (mut sdata, mut exchange, mut success) = match auth.begin(mechanism, &cbuf, tls) {
            BeginOutcome::Challenge { data, exchange } => (data, Some(exchange), None),
            BeginOutcome::Success {
                final_data,
                success,
            } => (final_data.unwrap_or_default(), None, Some(success)),
            BeginOutcome::Failed(_) => return Mediated::Denied,
        };

        loop {
            if let Some(s) = success.take() {
                // Deliver any trailing server data (SCRAM server-final) so the client can verify.
                if !sdata.is_empty() && cstate == ClientState::Running {
                    let mut tmp = Vec::new();
                    let _ = client.step(Some(&sdata), &mut tmp);
                }
                return Mediated::Ok(s);
            }
            // Feed the server challenge to the client; relay its response back to the server.
            cbuf.clear();
            cstate = match client.step(Some(&sdata), &mut cbuf) {
                Ok(s) => s,
                Err(_) => return Mediated::Denied,
            };
            let ex = match exchange.as_mut() {
                Some(ex) => ex,
                None => return Mediated::Denied,
            };
            match ex.step(&cbuf) {
                StepOutcome::Challenge(data) => sdata = data,
                StepOutcome::Success {
                    final_data,
                    success: s,
                } => {
                    sdata = final_data.unwrap_or_default();
                    success = Some(s);
                }
                StepOutcome::Failed(_) => return Mediated::Denied,
            }
        }
    }

    #[test]
    fn scram_sha256_round_trip_against_rsasl_client() {
        let store = store_with_user("alice", "correct horse", &[Role::Operator]);
        let auth = Authenticator::new(store.clone());
        match mediate(
            &auth,
            MECH_SCRAM_SHA_256,
            "alice",
            "correct horse",
            TlsState::plaintext(),
        ) {
            Mediated::Ok(s) => {
                assert_eq!(s.principal.username, "alice");
                assert!(s.principal.has(Capability::SessionSeeAll));
                // The token resolves back to the same user (minted on success).
                let p = store.principal_for_token(&s.token).expect("token resolves");
                assert_eq!(p.user_id, s.principal.user_id);
                assert!(s
                    .principal_view
                    .capabilities
                    .contains(&"session_write".to_string()));
            }
            Mediated::Denied => panic!("correct SCRAM credentials must authenticate"),
        }
    }

    #[test]
    fn scram_wrong_password_and_unknown_user_both_denied_no_oracle() {
        let store = store_with_user("alice", "correct horse", &[Role::User]);
        let auth = Authenticator::new(store);
        let wrong = mediate(
            &auth,
            MECH_SCRAM_SHA_256,
            "alice",
            "nope",
            TlsState::plaintext(),
        );
        let ghost = mediate(
            &auth,
            MECH_SCRAM_SHA_256,
            "ghost",
            "nope",
            TlsState::plaintext(),
        );
        assert!(matches!(wrong, Mediated::Denied), "wrong password denied");
        assert!(matches!(ghost, Mediated::Denied), "unknown user denied");
    }

    #[test]
    fn token_minted_only_on_successful_scram() {
        let store = store_with_user("alice", "pw-correct", &[Role::User]);
        let auth = Authenticator::new(store.clone());
        assert_eq!(store.session_count().unwrap(), 0);
        // A failed exchange mints nothing.
        let _ = mediate(
            &auth,
            MECH_SCRAM_SHA_256,
            "alice",
            "pw-wrong",
            TlsState::plaintext(),
        );
        assert_eq!(store.session_count().unwrap(), 0, "no token on failure");
        // A successful exchange mints exactly one.
        let _ = mediate(
            &auth,
            MECH_SCRAM_SHA_256,
            "alice",
            "pw-correct",
            TlsState::plaintext(),
        );
        assert_eq!(store.session_count().unwrap(), 1, "one token on success");
    }

    #[test]
    fn plain_verifies_over_tls_and_is_refused_without_tls() {
        let store = store_with_user("bob", "hunter2", &[Role::User]);
        let auth = Authenticator::new(store);
        // Over TLS, correct password authenticates.
        let tls = TlsState {
            is_tls: true,
            peer_cert_fingerprint: None,
        };
        assert!(matches!(
            mediate(&auth, MECH_PLAIN, "bob", "hunter2", tls.clone()),
            Mediated::Ok(_)
        ));
        // Over TLS, wrong password denied.
        assert!(matches!(
            mediate(&auth, MECH_PLAIN, "bob", "wrong", tls),
            Mediated::Denied
        ));
        // On a plaintext transport PLAIN is not even advertised; begin refuses it.
        assert!(matches!(
            mediate(&auth, MECH_PLAIN, "bob", "hunter2", TlsState::plaintext()),
            Mediated::Denied
        ));
    }

    #[test]
    fn plain_not_advertised_without_tls_but_scram_is() {
        let store = store_with_user("bob", "pw", &[Role::User]);
        let auth = Authenticator::new(store);
        let plain = auth.advertised_mechanisms(&TlsState::plaintext());
        assert_eq!(plain, vec![MECH_SCRAM_SHA_256.to_string()]);
        let tls = auth.advertised_mechanisms(&TlsState {
            is_tls: true,
            peer_cert_fingerprint: None,
        });
        assert!(tls.contains(&MECH_PLAIN.to_string()));
        assert!(tls.contains(&MECH_EXTERNAL.to_string()));
        assert_eq!(tls.first().map(String::as_str), Some(MECH_SCRAM_SHA_256));
    }

    #[test]
    fn external_unmapped_fingerprint_denies() {
        let store = store_with_user("svc", "pw", &[Role::Operator]);
        let auth = Authenticator::new(store);
        // A presented (but unmapped, since the table is not yet wired) client cert denies.
        let tls = TlsState {
            is_tls: true,
            peer_cert_fingerprint: Some("ab12cd34".into()),
        };
        match auth.begin(MECH_EXTERNAL, b"", tls) {
            BeginOutcome::Failed(_) => {}
            _ => panic!("unmapped EXTERNAL certificate must deny (fail-closed)"),
        }
        // EXTERNAL with no presented client cert also denies.
        let tls_no_cert = TlsState {
            is_tls: true,
            peer_cert_fingerprint: None,
        };
        assert!(matches!(
            auth.begin(MECH_EXTERNAL, b"", tls_no_cert),
            BeginOutcome::Failed(_)
        ));
    }

    #[test]
    fn external_mapped_fingerprint_authenticates() {
        let store = store_with_user("svc", "pw", &[Role::Operator]);
        // Enroll the verified client-cert fingerprint -> user (the store-level writer).
        let user = store.find_user("svc").unwrap().unwrap();
        store
            .set_external_identity(&user.id, "ab12cd34")
            .expect("enroll fingerprint");
        let auth = Authenticator::new(store);
        let tls = TlsState {
            is_tls: true,
            peer_cert_fingerprint: Some("ab12cd34".into()),
        };
        match auth.begin(MECH_EXTERNAL, b"", tls) {
            BeginOutcome::Success { success, .. } => {
                assert_eq!(success.principal.username, "svc");
                assert!(success.principal.has(Capability::SessionControlAny));
            }
            _ => panic!("a mapped EXTERNAL certificate must authenticate"),
        }
        // A different, unmapped fingerprint still denies (fail-closed).
        let auth2 = {
            let store = store_with_user("svc2", "pw", &[Role::Operator]);
            Authenticator::new(store)
        };
        let tls_unknown = TlsState {
            is_tls: true,
            peer_cert_fingerprint: Some("ab12cd34".into()),
        };
        assert!(matches!(
            auth2.begin(MECH_EXTERNAL, b"", tls_unknown),
            BeginOutcome::Failed(_)
        ));
    }

    #[test]
    fn unknown_mechanism_is_rejected() {
        let store = store_with_user("alice", "pw", &[Role::User]);
        let auth = Authenticator::new(store);
        assert!(matches!(
            auth.begin("GSSAPI", b"", TlsState::plaintext()),
            BeginOutcome::Failed(_)
        ));
        assert!(matches!(
            auth.begin("NONSENSE!!", b"", TlsState::plaintext()),
            BeginOutcome::Failed(_)
        ));
    }

    #[test]
    fn resume_resolves_token_and_rejects_invalid() {
        let store = store_with_user("carol", "pw", &[Role::Operator]);
        let auth = Authenticator::new(store.clone());
        let user = store.find_user("carol").unwrap().unwrap();
        let token = store
            .mint_session(&user.id, DEFAULT_SESSION_TTL_SECS, "scram-sha-256")
            .unwrap();
        match auth.resume(&token) {
            BeginOutcome::Success { success, .. } => {
                assert_eq!(success.principal.username, "carol");
                assert_eq!(success.token, token);
            }
            _ => panic!("valid token must resume"),
        }
        // Unknown token rejected.
        assert!(matches!(auth.resume("deadbeef"), BeginOutcome::Failed(_)));
        // Revoked token rejected.
        store.revoke_token(&token).unwrap();
        assert!(matches!(auth.resume(&token), BeginOutcome::Failed(_)));
    }

    #[test]
    fn resume_rejects_expired_and_disabled() {
        let store = store_with_user("dave", "pw", &[Role::User]);
        let auth = Authenticator::new(store.clone());
        let user = store.find_user("dave").unwrap().unwrap();
        // Expired token (negative TTL).
        let expired = store.mint_session(&user.id, -1, "scram-sha-256").unwrap();
        assert!(matches!(auth.resume(&expired), BeginOutcome::Failed(_)));
        // A live token for a then-disabled user is rejected (disable revokes sessions).
        let live = store
            .mint_session(&user.id, DEFAULT_SESSION_TTL_SECS, "scram-sha-256")
            .unwrap();
        store.set_disabled(&user.id, true).unwrap();
        assert!(matches!(auth.resume(&live), BeginOutcome::Failed(_)));
    }
}
