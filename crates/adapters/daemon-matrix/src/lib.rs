// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-matrix` — the Matrix chat transport (see `daemon-matrix-transport-spec.md`).
//!
//! An in-process `NodeApi`-client adapter, structurally the **inverse of `daemon-acp`**: it isolates
//! the heavy `matrix-sdk` deps and drives *our* engine as a client. One instance hosts N accounts;
//! each account is a transport instance (`matrix/@bot:hs.org`) with its own client, on-disk E2EE
//! store, and sync loop. The host owns routing: the adapter normalises a room event into an `Origin`
//! and hands it to the reusable `daemon-ingest` gate (inbound); replies flow back through a
//! `daemon-delivery` `Projector` to the session's `Primary` room (outbound). No matrix-sdk/ruma type
//! leaves this crate.

#![forbid(unsafe_code)]
// matrix-sdk's deeply-nested instrumented async futures (e.g. `Client::sync`) overflow the default
// auto-trait (`Send`) evaluation recursion limit when spawned; raise it (matrix-sdk's own guidance).
#![recursion_limit = "512"]
// Phase 4: test code may use raw fs/reqwest/Command; the --lib pass still guards production.
#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]

mod account;
pub mod adapter;
mod auth;
pub mod config;
mod inbound;
mod invite;
mod login;
mod mapping;
mod membership;
mod outbound;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use matrix_sdk::config::SyncSettings;
use matrix_sdk::store::RoomLoadSettings;
use matrix_sdk::Client;

use daemon_api::NodeApi;
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

pub use account::{Account, StoredSession};
pub use adapter::MatrixAdapter;
pub use auth::{sso_begin, sso_complete, MatrixAuthFlowFactory, MatrixLogin, SsoSession};
pub use config::{MatrixConfig, MatrixRoute};
pub use inbound::{on_room_message, InboundCtx};
pub use invite::{on_stripped_member, InviteCtx};
pub use login::login;
pub use outbound::{DeliveryManager, MatrixProjector};

use account::{account_store_dir, bare_account, build_client};

/// The transport family this adapter provisions (`AccountProvisioning::bound_accounts`).
pub(crate) const FAMILY: &str = "matrix";

/// The shared registry of live, session-restored clients keyed by their instance-qualified
/// transport id (`matrix/@bot:hs.org`). Populated by [`serve`] at bring-up and read by the
/// `MessagingProtocol` feature-trait method bodies (which only hold `&self`, so they recover the
/// account's [`Client`] from here). A `tokio::sync::RwLock` (not std) so the read in an `async`
/// verb body never blocks the runtime, and writers (bring-up) don't hold a guard across an await.
pub(crate) type LiveClients = Arc<tokio::sync::RwLock<HashMap<TransportId, Client>>>;

/// Persist refreshed session tokens back to the credential subsystem (spec §6.2: the credential
/// store is authoritative over the SDK's session copy). matrix-sdk refreshes tokens in-memory (the
/// client is built with `handle_refresh_tokens`); this polls the session and writes it back on
/// change via `AccountProvisioning::store_account_credential`.
fn spawn_session_writeback(
    client: Client,
    provisioning: Arc<dyn AccountProvisioning>,
    credential_ref: String,
    homeserver: String,
) {
    tokio::spawn(async move {
        let mut last: Option<String> = None;
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            let Some(session) = client.matrix_auth().session() else {
                continue;
            };
            let stored = StoredSession {
                homeserver: homeserver.clone(),
                session,
            };
            let blob = match stored.to_blob() {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(error = %e, "matrix: serializing session for write-back");
                    continue;
                }
            };
            if last.as_deref() == Some(blob.as_str()) {
                continue;
            }
            match provisioning.store_account_credential(&credential_ref, &blob) {
                Ok(()) => last = Some(blob),
                Err(e) => tracing::warn!(error = %e, "matrix: session write-back failed"),
            }
        }
    });
}

/// Bring up every credential-bound Matrix account and run the inbound + outbound loops until each
/// account's sync ends (or the task is aborted). Spawned in-process at host launch (next to the HTTP
/// surface). `provisioning` is the host's in-process `AccountProvisioning` seam (the same node that
/// backs `api`); enumeration + secret resolution stay in-process (no wire crossing).
pub async fn serve(
    api: Arc<dyn NodeApi>,
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: MatrixConfig,
    live_clients: LiveClients,
    sink: Option<Arc<dyn daemon_api::LifecycleSink>>,
) {
    if !cfg.enabled {
        return;
    }
    let accounts = provisioning.bound_accounts(FAMILY);
    if accounts.is_empty() {
        tracing::info!("matrix: enabled but no bound matrix accounts; nothing to do");
        return;
    }

    let ingestor = Arc::new(daemon_ingest::Ingestor::with_policy(
        api.clone(),
        cfg.ingest_policy(),
    ));
    let routes = Arc::new(cfg.routes.clone());

    let mut clients: HashMap<TransportId, Client> = HashMap::new();
    let mut brought_up: Vec<Account> = Vec::new();

    for acct in &accounts {
        let Some(blob) = provisioning.account_credential(&acct.credential_ref) else {
            tracing::warn!(
                instance = %acct.transport_instance.as_str(),
                credential_ref = %acct.credential_ref,
                "matrix: no stored session for account; run `daemon matrix login` first — skipping"
            );
            continue;
        };
        let stored = match StoredSession::from_blob(&blob) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, credential_ref = %acct.credential_ref, "matrix: bad session blob; skipping");
                continue;
            }
        };
        let store_dir = account_store_dir(&cfg.store_root, &acct.credential_ref);
        let client = match build_client(&stored.homeserver, &store_dir).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, instance = %acct.transport_instance.as_str(), "matrix: client build failed; skipping");
                continue;
            }
        };
        if let Err(e) = client
            .matrix_auth()
            .restore_session(stored.session.clone(), RoomLoadSettings::default())
            .await
        {
            tracing::warn!(error = %e, instance = %acct.transport_instance.as_str(), "matrix: restore_session failed; skipping");
            continue;
        }
        let Some(me) = client.user_id().map(|u| u.to_owned()) else {
            tracing::warn!(instance = %acct.transport_instance.as_str(), "matrix: restored client has no user id; skipping");
            continue;
        };

        spawn_session_writeback(
            client.clone(),
            provisioning.clone(),
            acct.credential_ref.clone(),
            stored.homeserver.clone(),
        );

        clients.insert(acct.transport_instance.clone(), client.clone());
        brought_up.push(Account {
            transport: acct.transport_instance.clone(),
            bare: bare_account(&acct.transport_instance).to_string(),
            client,
        });
        tracing::info!(instance = %acct.transport_instance.as_str(), user = %me, "matrix: account brought up");
    }

    if brought_up.is_empty() {
        tracing::warn!("matrix: no accounts could be brought up; exiting");
        return;
    }

    // Publish the live clients so the adapter's `SupportsConversations`/`SupportsMembership` method
    // bodies (which only have `&self`) can resolve the per-account `Client` to execute management
    // verbs against (send / set_topic / create / m.room.member invite·kick·ban·power-levels).
    {
        let mut guard = live_clients.write().await;
        for acct in &brought_up {
            guard.insert(acct.transport.clone(), acct.client.clone());
        }
    }

    let projector = Arc::new(MatrixProjector::new(
        api.clone(),
        ingestor.clone(),
        clients.clone(),
    ));
    let delivery = Arc::new(DeliveryManager::new(api.clone(), projector));

    // Resume delivery for any sessions this transport already owns (reconnect / restart),
    // walking the wire pages so every owned session resumes (not just the first page).
    for acct in &brought_up {
        let mut after: Option<String> = None;
        loop {
            let page = api
                .delivery_sessions(acct.transport.clone(), after.take())
                .await;
            for session in page.items {
                delivery.ensure(session, acct.transport.clone());
            }
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
    }

    // Register the inbound handler + start each account's sync loop.
    let mut sync_tasks = Vec::new();
    for acct in brought_up {
        let me = match acct.client.user_id() {
            Some(u) => u.to_owned(),
            None => continue,
        };
        let ctx = InboundCtx {
            ingestor: ingestor.clone(),
            delivery: delivery.clone(),
            routes: routes.clone(),
            bare: acct.bare.clone(),
            transport: acct.transport.clone(),
            me: me.clone(),
        };
        // Invite acceptance (EIO-11): join rooms this account is invited to (policy-gated), so an
        // externally-invited bot lands in the room and its rooms list / `ConvList` reflect it.
        acct.client.add_event_handler_context(InviteCtx {
            me,
            transport: acct.transport.clone(),
            auto_accept: cfg.auto_accept_invites,
        });
        acct.client.add_event_handler(on_stripped_member);
        acct.client.add_event_handler_context(ctx);
        acct.client.add_event_handler(on_room_message);
        // [waveA:node-v30] membership push: report conversation/membership deltas through the node
        // lifecycle sink (item 3). Re-derives `me` from the client so it does not depend on the
        // move order of the surrounding ctx registrations.
        if let Some(sink) = &sink {
            if let Some(me) = acct.client.user_id().map(|u| u.to_owned()) {
                acct.client
                    .add_event_handler_context(crate::membership::MembershipCtx {
                        sink: sink.clone(),
                        transport: acct.transport.clone(),
                        me,
                    });
                acct.client
                    .add_event_handler(crate::membership::on_room_member);
            }
        }

        let client = acct.client.clone();
        let instance = acct.transport.clone();
        sync_tasks.push(tokio::spawn(async move {
            tracing::info!(instance = %instance.as_str(), "matrix: sync loop started");
            if let Err(e) = client.sync(SyncSettings::default()).await {
                tracing::warn!(error = %e, instance = %instance.as_str(), "matrix: sync loop ended");
            }
        }));
    }

    for task in sync_tasks {
        let _ = task.await;
    }
}
