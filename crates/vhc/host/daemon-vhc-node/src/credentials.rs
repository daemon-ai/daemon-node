// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Node-side credential authorship + per-run identity provisioning (D-P8; architecture §6.3.1).
//!
//! The NODE — never the worker subprocess — mints each run instance's signing identity and
//! authors the plane-selection credentials the worker attaches with:
//!
//! 1. **Per-run identity**: mint the CSPRNG per-run key for `(run, role, incarnation)`, issue its
//!    [`RunKeyCertificate`] under the node's durable **base identity**, and persist both in the
//!    identity keystore. The base identity never leaves the node process — the worker resolves the
//!    key + certificate READ-ONLY by reference (it never mints, never reads `base.key`).
//! 2. **Credentials**: author a secrets-free [`SessionCredentials`] plane selection (the WS base,
//!    presign base, public bootstrap material) and — when the registry auth is secret-bearing —
//!    write the token into a keystore CREDENTIALS RECORD, carrying only its `secret_ref` on the
//!    wire ([CI-9]: token material never rides the command payload or a journal).
//!
//! The authored bytes are what the node hands `JoinRun.credentials`; the returned `credentials_ref`
//! is what `vhc.db` persists (never the secret itself).

use daemon_vhc_proto::RunKeyCertificate;
use daemon_vhc_session::config::{RegistryAuthConfig, RegistryConfig};
use daemon_vhc_session::keystore::{KeystoreError, VhcKeystore};
use daemon_vhc_session::protocol::{
    CheckpointRestore, CredentialsRecord, IrohPlane, SessionCredentials, WsAuthSpec,
};
use daemon_vhc_session::provisioning::{provision_run_identity, ProvisionScope};

use crate::service::VhcError;

/// The coordinator-seat bootstrap a trainer's credentials carry: the incumbent's certificate (so
/// the worker's attach can authenticate coordinator frames before the on-plane §12.3 distribution
/// arrives) and the seat-published control endpoint (dialed in preference to the discovered base).
#[derive(Clone, Default)]
pub struct SeatBootstrap {
    /// The incumbent coordinator's certificate (chained to a genesis-trusted base; the worker's
    /// `CertCheck` still gates trust — an untrusted cert simply never authenticates a frame).
    pub peer_certs: Vec<RunKeyCertificate>,
    /// The seat-published WS control endpoint, when the lease carries one.
    pub ws_base: Option<String>,
}

/// The output of one authorship pass: the wire credentials bytes for `JoinRun.credentials` and
/// the keystore reference `vhc.db` persists (`None` when no secret record was needed).
pub struct AuthoredJoin {
    /// The canonical-CBOR [`SessionCredentials`] bytes (secrets-free).
    pub wire: Vec<u8>,
    /// The credentials-record reference (`None` for an unauthenticated target).
    pub credentials_ref: Option<String>,
}

/// The run-instance identity the node provisions + authors credentials for.
pub struct RunInstanceIdentity<'a> {
    /// The run label (keystore + WS route key).
    pub run_label: &'a str,
    /// The run's cryptographic identity (genesis hash).
    pub genesis_hash: [u8; 32],
    /// The transition-chain epoch (0 at first join).
    pub epoch: u64,
    /// The envelope role label this instance runs.
    pub role: &'a str,
    /// The node-minted, never-reused incarnation.
    pub incarnation: u64,
    /// The pinned module hash (from the admitted tuple).
    pub module_hash: [u8; 32],
}

/// Provision the per-run identity and author the join credentials for `identity`, resolving the
/// WS/presign endpoints from `coordinator` + the registry config. Idempotent within an
/// incarnation: the keystore recovers the same key on a re-author, and the certificate is
/// re-issued over it (a crash-safe resume mints nothing new).
///
/// # Errors
/// A keystore / certificate-issuance failure ([`VhcError::Internal`]).
pub fn author_join(
    keystore: &VhcKeystore,
    identity: &RunInstanceIdentity<'_>,
    coordinator: &str,
    registry: &RegistryConfig,
    restore: Option<CheckpointRestore>,
    local_payload_plane: bool,
    seat: SeatBootstrap,
    iroh: Option<IrohPlane>,
) -> Result<AuthoredJoin, VhcError> {
    let cred = |e: KeystoreError| VhcError::Internal(format!("credential authorship: {e}"));

    // -- per-run identity: mint the key + issue its certificate under the base identity ---------
    provision_run_identity(
        keystore,
        &ProvisionScope {
            run_label: identity.run_label,
            genesis_hash: identity.genesis_hash,
            epoch: identity.epoch,
            role: identity.role,
            incarnation: identity.incarnation,
            module_hash: identity.module_hash,
        },
    )
    .map_err(|e| VhcError::Internal(format!("provision run identity: {e}")))?;

    // -- credentials: secrets to the keystore record, only a reference on the wire --------------
    let ws_auth = registry_ws_auth(&registry.auth);
    let (secret_ref, expires_at_ms) = match ws_auth {
        WsAuthSpec::None => (None, 0),
        secret => {
            let reference = keystore
                .store_run_credentials(
                    identity.run_label,
                    identity.role,
                    identity.incarnation,
                    &CredentialsRecord {
                        ws_auth: secret,
                        expires_at_ms: 0,
                    },
                )
                .map_err(cred)?;
            (Some(reference), 0)
        }
    };

    // Plane selection: a configured shared FILESYSTEM payload root (`[vhc] payload_dir`, the
    // multi-node single-host topology) uses the fs content store — the worker roots it at the
    // node-delivered shared dir, so `presign_base` stays absent even when a registry base is
    // configured for discovery + the seat CAS. Otherwise the presigned R2 plane is selected
    // from the registry base.
    let presign_base =
        (!local_payload_plane && !registry.base.is_empty()).then(|| registry.base.clone());
    // The seat-published endpoint (when present) is the authoritative coordinator control plane;
    // otherwise the node-resolved discovery coordinator.
    let ws_base = seat
        .ws_base
        .filter(|b| !b.is_empty())
        .or_else(|| (!coordinator.is_empty()).then(|| coordinator.to_string()));
    let credentials = SessionCredentials {
        genesis_hash: identity.genesis_hash,
        ws_base,
        // The secret-bearing auth lives in the keystore record (secret_ref); the wire body's
        // auth stays None ([CI-9] — never a token on the command payload).
        ws_auth: WsAuthSpec::None,
        // The node-resolved iroh plane (registry-served signed roster, verified node-side;
        // `None` = WS-only — the iroh plane is opt-in via `[vhc].iroh.enabled`).
        iroh,
        presign_base,
        // Bootstrap trust: the incumbent coordinator's certificate from the seat lease, so the
        // worker authenticates coordinator frames without waiting for the on-plane announcement
        // (which a late subscriber misses). `CertCheck` still gates trust by genesis base.
        peer_certs: seat.peer_certs,
        secret_ref,
        expires_at_ms,
        restore,
    };
    let wire = credentials
        .to_bytes()
        .map_err(|e| VhcError::Internal(format!("encode session credentials: {e}")))?;
    Ok(AuthoredJoin {
        wire,
        credentials_ref: credentials.secret_ref,
    })
}

/// Map the node's registry auth config onto the worker-wire auth vocabulary.
fn registry_ws_auth(auth: &RegistryAuthConfig) -> WsAuthSpec {
    match auth {
        RegistryAuthConfig::None => WsAuthSpec::None,
        RegistryAuthConfig::Bearer { token } => WsAuthSpec::Bearer(token.clone()),
        RegistryAuthConfig::Internal { org_id, actor } => WsAuthSpec::Internal {
            org_id: org_id.clone(),
            actor: actor.clone(),
        },
    }
}
