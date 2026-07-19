// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Live transport-provider construction — the node-authored plane selection
//! ([`SessionCredentials`]) becomes the [`RoleProviders`] a role session binds (architecture
//! §7.1; ABI §12.6).
//!
//! Selection rules (all fail closed — a plane that cannot be built refuses the join typed,
//! never a silent local run):
//!
//! - **Control plane**: `WsControlPlane` against `ws_base` (else the node-resolved,
//!   allowlist-checked `JoinRun.coordinator`); the session's own §12.3 certificate announcement
//!   is registered as a resubscribe frame so every reconnect re-announces. With an iroh half
//!   present, the WS and gossip planes compose through `DualPlane` (cross-plane dedupe); the
//!   iroh transport SECRET comes from the identity keystore by reference, never the wire.
//! - **Payload + artifact planes**: `presign_base` present ⇒ the presigned R2 content store
//!   (`payload/<blake3>` under the run's namespace); absent ⇒ the filesystem content store under
//!   the run's node-delivered state dir (`<run state dir>/payload/`). Both are content-addressed
//!   ([RS-4]): the module names content, the pump re-verifies every fetched object.

use std::sync::Arc;

use daemon_vhc_net::{
    ContentStore, ControlPlane, FsContentStore, HttpPresignClient, R2Store, ReconnectConfig, RunId,
    WsAuth, WsConfig, WsControlPlane,
};

use crate::journal_home;
use crate::keystore::VhcKeystore;
use crate::protocol::{SessionCredentials, WsAuthSpec};
use crate::role_session::RoleProviders;

/// Everything the live attach needs beyond the credentials body itself.
pub struct LiveAttachInputs<'a> {
    /// The node-authored plane selection (from `JoinRun.credentials`).
    pub credentials: &'a SessionCredentials,
    /// The node-resolved coordinator base URL (`JoinRun.coordinator`) — the WS fallback when the
    /// credentials carry no explicit `ws_base`.
    pub coordinator: &'a str,
    /// The run label (the WS route key + the R2 store's run namespace).
    pub run_label: &'a str,
    /// The session's own §12.3 certificate announcement bytes (published on attach by the
    /// session; registered here as a WS resubscribe frame so reconnects re-announce).
    pub own_cert_announcement: Vec<u8>,
    /// The identity keystore (the iroh transport secret's home — resolved by reference).
    pub keystore: &'a VhcKeystore,
}

/// Build the live [`RoleProviders`] from a plane selection. Fails typed on any plane that
/// cannot be constructed.
///
/// # Errors
/// A human-readable refusal (the worker surfaces it as the typed join error).
pub async fn build_role_providers(inputs: LiveAttachInputs<'_>) -> Result<RoleProviders, String> {
    let creds = inputs.credentials;

    // Resolve the auth: the wire body carries no secret ([CI-9]); when a `secret_ref` names a
    // keystore credentials record, its `ws_auth` is the live token material (re-read here so an
    // expiry-driven node refresh reaches a fresh dial). Absent record ⇒ the body's own (public)
    // auth stands.
    let auth = match &creds.secret_ref {
        Some(reference) => inputs
            .keystore
            .run_credentials_by_ref(inputs.run_label, reference)
            .map_err(|e| format!("resolve credentials record `{reference}`: {e}"))?
            .map(|record| record.ws_auth)
            .unwrap_or_else(|| creds.ws_auth.clone()),
        None => creds.ws_auth.clone(),
    };

    // -- control plane: WS (mandatory), composed with iroh gossip when selected ---------------
    let ws_base = match &creds.ws_base {
        Some(base) if !base.is_empty() => base.clone(),
        _ if !inputs.coordinator.is_empty() => inputs.coordinator.to_string(),
        _ => {
            return Err(
                "live attach selected but no WS control-plane base (neither the credentials' \
                 ws_base nor JoinRun.coordinator names one)"
                    .into(),
            )
        }
    };
    let ws = WsControlPlane::connect(WsConfig {
        base_url: ws_base.clone(),
        run_id: inputs.run_label.to_string(),
        auth: ws_auth(&auth),
        reconnect: ReconnectConfig::default(),
    })
    .await
    .map_err(|e| format!("ws control plane {ws_base}: {e}"))?;
    ws.add_resubscribe_frame(inputs.own_cert_announcement);
    let ws = Arc::new(ws);

    let control: Arc<dyn ControlPlane> = match &creds.iroh {
        None => ws,
        Some(iroh) => {
            #[cfg(feature = "live-iroh")]
            {
                let gossip = connect_iroh(iroh, creds.genesis_hash, inputs.keystore).await?;
                Arc::new(daemon_vhc_net::DualPlane::pair(ws, gossip))
            }
            #[cfg(not(feature = "live-iroh"))]
            {
                let _ = (iroh, inputs.keystore);
                return Err(
                    "credentials select an iroh plane but this worker build carries no iroh \
                     transport (build with the networked feature set)"
                        .into(),
                );
            }
        }
    };

    // -- payload + artifact planes: content-addressed, R2 or the run's fs store ----------------
    let stores: Arc<dyn ContentStore> = match &creds.presign_base {
        Some(base) if !base.is_empty() => {
            let egress = daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
                .map_err(|e| format!("egress client: {e}"))?;
            let presign_egress =
                daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
                    .map_err(|e| format!("presign egress client: {e}"))?;
            let presign = presign_client(presign_egress, base, &auth);
            Arc::new(R2Store::new(presign, egress, RunId::new(inputs.run_label)))
        }
        _ => {
            let root = journal_home::run_dir_from_env().ok_or_else(|| {
                "no presign base and no run-state root reference (the fs content store needs \
                 the node-delivered run dir)"
                    .to_string()
            })?;
            let dir = journal_home::payload_dir(&root, inputs.run_label);
            Arc::new(
                FsContentStore::open(&dir)
                    .map_err(|e| format!("fs content store {}: {e}", dir.display()))?,
            )
        }
    };

    Ok(RoleProviders {
        control,
        payloads: stores.clone(),
        artifacts: stores,
    })
}

/// Map the credentials' auth vocabulary onto the WS client's.
fn ws_auth(spec: &WsAuthSpec) -> WsAuth {
    match spec {
        WsAuthSpec::None => WsAuth::None,
        WsAuthSpec::Bearer(token) => WsAuth::Bearer(token.clone()),
        WsAuthSpec::Internal { org_id, actor } => WsAuth::Internal {
            org_id: org_id.clone(),
            actor: actor.clone(),
        },
    }
}

/// An [`HttpPresignClient`] for `base` carrying the same credential the WS plane uses.
fn presign_client(
    egress: daemon_egress::EgressClient,
    base: &str,
    auth: &WsAuthSpec,
) -> HttpPresignClient {
    let client = HttpPresignClient::new(egress, base.to_string());
    match auth {
        WsAuthSpec::None => client,
        WsAuthSpec::Bearer(token) => client.with_bearer(token.clone()),
        WsAuthSpec::Internal { org_id, actor } => {
            client.with_internal(org_id.clone(), actor.clone())
        }
    }
}

/// Connect the iroh gossip half of a dual plane: the transport secret comes from the identity
/// keystore (its own key, distinct from every signing identity); the topic derives from the
/// genesis hash.
#[cfg(feature = "live-iroh")]
async fn connect_iroh(
    plane: &crate::protocol::IrohPlane,
    genesis_hash: [u8; 32],
    keystore: &VhcKeystore,
) -> Result<Arc<daemon_vhc_net::IrohGossip>, String> {
    use daemon_vhc_net::{IrohGossip, IrohGossipConfig, IrohPeer, RebroadcastConfig};

    let secret = keystore
        .iroh_secret()
        .map_err(|e| format!("iroh transport secret: {e}"))?;
    let roster = plane
        .roster
        .iter()
        .map(|p| IrohPeer {
            endpoint_id: p.endpoint_id,
            direct_addrs: p
                .direct_addrs
                .iter()
                .filter_map(|a| a.parse().ok())
                .collect(),
            relay_url: p.relay_url.clone(),
        })
        .collect();
    let gossip = IrohGossip::connect(IrohGossipConfig {
        secret_key: *secret.bytes(),
        relay_urls: plane.relay_urls.clone(),
        roster,
        genesis_hash,
        rebroadcast: RebroadcastConfig::default(),
        bind_addr: None,
    })
    .await
    .map_err(|e| format!("iroh gossip plane: {e}"))?;
    Ok(Arc::new(gossip))
}
