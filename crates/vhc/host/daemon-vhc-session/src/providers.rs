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
use std::time::Duration;

use daemon_vhc_net::{
    ContentStore, ControlPlane, FsContentStore, HttpPresignClient, R2Store, ReconnectConfig, RunId,
    WsAuth, WsConfig, WsControlPlane,
};

use crate::journal_home;
use crate::keystore::VhcKeystore;
use crate::protocol::{ErrorClass, SessionCredentials, WsAuthSpec};
use crate::role_session::RoleProviders;

/// The default **control-plane bring-up deadline**: how long one plane's connect/bind may take
/// before the attach is refused as a typed, retryable timeout.
///
/// Bring-up must be bounded on EVERY platform. The Windows fleet-smoke STOP was a worker that
/// idled ~11 minutes inside iroh endpoint bring-up: the node's `JoinRun` is one-way (the worker
/// streams its own events), so an attach that never returns is invisible — no typed error, no
/// terminal, no retry, and the assess deadline cannot help because the stall PRECEDES the timed
/// assess phase. A plane that cannot come up must fail fast and classify instead.
///
/// It is deliberately distinct from (and much shorter than) the compute-bound assess deadline
/// (`[vhc] assess_timeout_secs`, 300 s): bring-up is a dial + a socket bind, not device bring-up
/// and kernel compilation. `DAEMON_VHC_PLANE_BRINGUP_SECS` overrides it for slow/loaded lanes
/// (`0` is rejected — an unbounded bring-up is the defect).
pub const PLANE_BRINGUP_DEADLINE: Duration = Duration::from_secs(60);

/// The bring-up deadline in force: [`PLANE_BRINGUP_DEADLINE`], or the
/// `DAEMON_VHC_PLANE_BRINGUP_SECS` override when it parses to a non-zero second count.
fn plane_bringup_deadline() -> Duration {
    std::env::var("DAEMON_VHC_PLANE_BRINGUP_SECS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map_or(PLANE_BRINGUP_DEADLINE, Duration::from_secs)
}

/// Why a live attach could not build its providers — typed so the worker classifies the join
/// refusal (a transport fault is RETRYABLE; a bad plane selection is not) instead of collapsing
/// every attach failure onto the module class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaneErrorKind {
    /// The plane's bring-up exceeded [`plane_bringup_deadline`] — retryable.
    Timeout,
    /// The plane could not be connected/bound (dial refused, socket bind failed, provider
    /// construction failed) — retryable.
    Transport,
    /// The node-authored selection itself is unusable here (no control-plane base, no run-state
    /// root, an unparseable bind address, an unresolvable credentials record, a build without the
    /// selected transport). Re-dialing cannot help; the node must re-author.
    Selection,
}

/// A typed live-attach failure: which plane, which class, and the operator-facing detail.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{plane} plane: {detail}")]
pub struct PlaneError {
    /// The plane being brought up (`control`, `iroh`, `payload`).
    pub plane: &'static str,
    /// The failure class the worker classifies on.
    pub kind: PlaneErrorKind,
    /// Operator-facing detail (never branched on).
    pub detail: String,
}

impl PlaneError {
    /// A bring-up that exceeded the deadline.
    fn timeout(plane: &'static str, what: &str, deadline: Duration) -> Self {
        Self {
            plane,
            kind: PlaneErrorKind::Timeout,
            detail: format!(
                "{what} did not come up within {}s (bring-up deadline; \
                 DAEMON_VHC_PLANE_BRINGUP_SECS overrides)",
                deadline.as_secs()
            ),
        }
    }

    /// A plane that could not be connected/bound.
    fn transport(plane: &'static str, detail: impl Into<String>) -> Self {
        Self {
            plane,
            kind: PlaneErrorKind::Transport,
            detail: detail.into(),
        }
    }

    /// An unusable plane selection.
    fn selection(plane: &'static str, detail: impl Into<String>) -> Self {
        Self {
            plane,
            kind: PlaneErrorKind::Selection,
            detail: detail.into(),
        }
    }

    /// The worker error class this failure is reported under: a bring-up timeout or transport
    /// fault is [`ErrorClass::Transient`] (the node re-converges under its retry budget); an
    /// unusable selection is [`ErrorClass::Module`] (re-authorship, not a re-dial).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self.kind {
            PlaneErrorKind::Timeout | PlaneErrorKind::Transport => ErrorClass::Transient,
            PlaneErrorKind::Selection => ErrorClass::Module,
        }
    }
}

/// Run `fut` under `deadline`, mapping the two failure shapes onto [`PlaneError`].
async fn bounded<T, E: std::fmt::Display>(
    plane: &'static str,
    what: &str,
    deadline: Duration,
    fut: impl std::future::Future<Output = Result<T, E>>,
) -> Result<T, PlaneError> {
    match tokio::time::timeout(deadline, fut).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(PlaneError::transport(plane, format!("{what}: {e}"))),
        Err(_elapsed) => Err(PlaneError::timeout(plane, what, deadline)),
    }
}

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
/// cannot be constructed, and every control-plane bring-up is BOUNDED by
/// [`plane_bringup_deadline`] — a plane that neither comes up nor errors is a typed timeout, never
/// an idle worker.
///
/// # Errors
/// A [`PlaneError`] naming the plane, its class (the worker classifies the join refusal on it),
/// and the operator-facing detail.
pub async fn build_role_providers(
    inputs: LiveAttachInputs<'_>,
) -> Result<RoleProviders, PlaneError> {
    let creds = inputs.credentials;

    // Resolve the auth: the wire body carries no secret ([CI-9]); when a `secret_ref` names a
    // keystore credentials record, its `ws_auth` is the live token material (re-read here so an
    // expiry-driven node refresh reaches a fresh dial). Absent record ⇒ the body's own (public)
    // auth stands.
    let auth = match &creds.secret_ref {
        Some(reference) => inputs
            .keystore
            .run_credentials_by_ref(inputs.run_label, reference)
            .map_err(|e| {
                PlaneError::selection(
                    "control",
                    format!("resolve credentials record `{reference}`: {e}"),
                )
            })?
            .map(|record| record.ws_auth)
            .unwrap_or_else(|| creds.ws_auth.clone()),
        None => creds.ws_auth.clone(),
    };

    // -- control plane: WS (mandatory), composed with iroh gossip when selected ---------------
    let ws_base = match &creds.ws_base {
        Some(base) if !base.is_empty() => base.clone(),
        _ if !inputs.coordinator.is_empty() => inputs.coordinator.to_string(),
        _ => {
            return Err(PlaneError::selection(
                "control",
                "live attach selected but no WS control-plane base (neither the credentials' \
                 ws_base nor JoinRun.coordinator names one)",
            ))
        }
    };
    tracing::debug!(run = inputs.run_label, ws_base = %ws_base, "live attach: connecting WS control plane");
    let ws = bounded(
        "control",
        &format!("ws control plane {ws_base}"),
        plane_bringup_deadline(),
        WsControlPlane::connect(WsConfig {
            base_url: ws_base.clone(),
            run_id: inputs.run_label.to_string(),
            auth: ws_auth(&auth),
            reconnect: ReconnectConfig::default(),
        }),
    )
    .await?;
    ws.add_resubscribe_frame(inputs.own_cert_announcement);
    let ws = Arc::new(ws);
    tracing::debug!(
        run = inputs.run_label,
        "live attach: WS control plane connected"
    );

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
                return Err(PlaneError::selection(
                    "iroh",
                    "credentials select an iroh plane but this worker build carries no iroh \
                     transport (build with the networked feature set)",
                ));
            }
        }
    };

    // -- payload + artifact planes: content-addressed, R2 or the run's fs store ----------------
    let stores: Arc<dyn ContentStore> = match &creds.presign_base {
        Some(base) if !base.is_empty() => {
            tracing::debug!(run = inputs.run_label, presign_base = %base, "live attach: presigned R2 content plane");
            let egress =
                daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
                    .map_err(|e| PlaneError::transport("payload", format!("egress client: {e}")))?;
            let presign_egress = daemon_egress::EgressClient::new(
                daemon_egress::EgressConfig::default(),
            )
            .map_err(|e| PlaneError::transport("payload", format!("presign egress client: {e}")))?;
            let presign = presign_client(presign_egress, base, &auth);
            Arc::new(R2Store::new(presign, egress, RunId::new(inputs.run_label)))
        }
        _ => {
            // The node-delivered payload-plane override wins (a shared single-host root, so
            // multi-process peers serve each other's content); else the run's own state dir.
            let dir = match journal_home::payload_dir_from_env() {
                Some(shared) => journal_home::payload_dir(&shared, inputs.run_label),
                None => {
                    let root = journal_home::run_dir_from_env().ok_or_else(|| {
                        PlaneError::selection(
                            "payload",
                            "no presign base and no run-state root reference (the fs content store \
                             needs the node-delivered run dir)",
                        )
                    })?;
                    journal_home::payload_dir(&root, inputs.run_label)
                }
            };
            tracing::debug!(run = inputs.run_label, dir = %dir.display(), "live attach: filesystem content plane");
            Arc::new(FsContentStore::open(&dir).map_err(|e| {
                PlaneError::transport(
                    "payload",
                    format!("fs content store {}: {e}", dir.display()),
                )
            })?)
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
) -> Result<Arc<daemon_vhc_net::IrohGossip>, PlaneError> {
    use daemon_vhc_net::{IrohGossip, IrohGossipConfig, IrohPeer, RebroadcastConfig};

    let secret = keystore
        .iroh_secret()
        .map_err(|e| PlaneError::selection("iroh", format!("iroh transport secret: {e}")))?;
    let roster: Vec<IrohPeer> = plane
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
    // The node-pinned bind address (the socket the node already PUBLISHED in this peer's roster
    // record): binding anything else would advertise addresses no one can dial, so a bad pin is
    // a typed refusal, never a silent ephemeral fallback.
    let bind_addr =
        match &plane.bind_addr {
            Some(raw) => Some(raw.parse().map_err(|e| {
                PlaneError::selection("iroh", format!("iroh bind addr `{raw}`: {e}"))
            })?),
            None => None,
        };
    let roster_len = roster.len();
    // BOUNDED: the endpoint bind + relay handshake + topic subscribe must land inside the
    // bring-up deadline. A platform that neither binds nor errors (the Windows fleet-smoke silent
    // hang) is a typed, retryable timeout here — never an idle worker holding an admitted ledger.
    let gossip = bounded(
        "iroh",
        "iroh gossip plane",
        plane_bringup_deadline(),
        IrohGossip::connect(IrohGossipConfig {
            secret_key: *secret.bytes(),
            relay_urls: plane.relay_urls.clone(),
            roster,
            genesis_hash,
            rebroadcast: RebroadcastConfig::default(),
            bind_addr,
        }),
    )
    .await?;
    tracing::info!(
        endpoint = %daemon_vhc_proto::Hash(gossip.node_id()).to_hex(),
        roster = roster_len,
        relays = plane.relay_urls.len(),
        "iroh gossip plane up"
    );
    Ok(Arc::new(gossip))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bring-up deadline is FINITE by default and comfortably clear of the assess deadline
    /// (`[vhc] assess_timeout_secs`, 300 s) it must not be confused with.
    #[test]
    fn the_bring_up_deadline_is_finite_and_distinct_from_assess() {
        assert!(PLANE_BRINGUP_DEADLINE > Duration::ZERO);
        assert!(
            PLANE_BRINGUP_DEADLINE < Duration::from_secs(300),
            "plane bring-up is a dial + a bind, not a compute-bound assess"
        );
    }

    /// Classification is what makes a stalled/refused transport RECOVERABLE: the node re-converges
    /// a `Transient` refusal under its retry budget, while an unusable selection stays a module
    /// class (re-authorship, not a re-dial).
    #[test]
    fn transport_and_timeout_classify_retryable_selection_does_not() {
        let timeout = PlaneError::timeout("iroh", "iroh gossip plane", Duration::from_secs(60));
        assert_eq!(timeout.class(), ErrorClass::Transient);
        assert!(
            timeout.detail.contains("60s"),
            "the refusal names the deadline it hit: {}",
            timeout.detail
        );
        assert_eq!(
            PlaneError::transport("control", "dial refused").class(),
            ErrorClass::Transient
        );
        assert_eq!(
            PlaneError::selection("iroh", "no iroh transport in this build").class(),
            ErrorClass::Module
        );
    }

    /// A bring-up that never completes is a typed timeout, not an indefinite await — the Windows
    /// silent-hang shape, reproduced with a future that never resolves.
    #[tokio::test]
    async fn a_never_completing_bring_up_times_out_typed() {
        let never = async {
            std::future::pending::<()>().await;
            Ok::<(), String>(())
        };
        let err = bounded(
            "iroh",
            "iroh gossip plane",
            Duration::from_millis(50),
            never,
        )
        .await
        .expect_err("a bring-up that never completes must not await forever");
        assert_eq!(err.kind, PlaneErrorKind::Timeout);
        assert_eq!(err.class(), ErrorClass::Transient);
        assert_eq!(err.plane, "iroh");
    }

    /// A bring-up that fails fast keeps its own detail and is still classified retryable (the
    /// macOS fleet-smoke shape: `endpoint bind failed: Failed to bind sockets`).
    #[tokio::test]
    async fn a_failing_bring_up_is_a_typed_transport_refusal() {
        let err = bounded("iroh", "iroh gossip plane", PLANE_BRINGUP_DEADLINE, async {
            Err::<(), String>("endpoint bind failed: Failed to bind sockets".to_string())
        })
        .await
        .expect_err("a failed bind is a refusal");
        assert_eq!(err.kind, PlaneErrorKind::Transport);
        assert!(err.detail.contains("Failed to bind sockets"), "{err}");
    }
}
