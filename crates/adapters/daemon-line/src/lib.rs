// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-line` — the LINE chat transport (Phase 1 of the multi-protocol messaging plan).
//!
//! An in-process `NodeApi`-client adapter, a sibling of `daemon-matrix`: it isolates the
//! `line-bot-sdk-rust` Messaging-API SDK and drives *our* engine as a client. One instance hosts N
//! LINE channels; each channel is a transport instance (`line/<handle>`) with its own bot client
//! (channel access token) and channel secret. Inbound LINE **webhook** events are verified +
//! normalised into an `Origin` and handed to the reusable `daemon-ingest` gate; replies flow back
//! through a `daemon-delivery` `Projector` as LINE **push** messages. No `line-bot-sdk-rust` type
//! leaves this crate.
//!
//! ## Bot-only
//!
//! There is no mature Rust LINE *user* client, so the sole auth mode is **bot** — a channel access
//! token plus channel secret (see [`auth`]). Membership administration (invite/kick/ban/roles) is not
//! a LINE bot capability and is therefore not exposed (see [`adapter`]).
//!
//! ## Inbound is webhook-push (Phase 2 must connect the ingress)
//!
//! Unlike Matrix's long-poll sync loop, LINE pushes events to a public HTTP endpoint. The inbound
//! seam is an adapter-owned axum router ([`inbound::webhook_router`]): `POST {webhook_path}/{handle}`,
//! signature-verified per account. [`serve`] binds + serves it itself when
//! [`LineConfig::webhook_bind`] is set. **Phase 2** must expose that path on a public URL, register it
//! as the channel's webhook in the LINE console, and either set `webhook_bind` (adapter-owned) or
//! mount [`inbound::webhook_router`] into a shared node ingress.

#![forbid(unsafe_code)]

pub mod account;
pub mod adapter;
pub mod auth;
pub mod config;
pub mod inbound;
mod mapping;
pub mod outbound;

use std::collections::HashMap;
use std::sync::Arc;

use daemon_api::NodeApi;
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

pub use account::{bare_account, derive_handle, LineAccount, StoredCredential};
pub use adapter::LineAdapter;
pub use auth::LineAuthFlowFactory;
pub use config::{LineConfig, LineRoute};
pub use inbound::{
    handle_webhook, serve_webhook, verify_and_parse, webhook_router, InboundMessage,
    WebhookAccount, WebhookState,
};
pub use outbound::{DeliveryManager, LineProjector};

/// The transport family this adapter provisions (`AccountProvisioning::bound_accounts`).
pub(crate) const FAMILY: &str = "line";

/// The shared registry of live, credential-restored bot clients keyed by their instance-qualified
/// transport id (`line/<handle>`). Populated by [`serve`] at bring-up and read by the
/// `MessagingProtocol` feature-trait method bodies (which only hold `&self`). A `tokio::sync::RwLock`
/// so the read in an `async` verb body never blocks the runtime.
pub type LiveClients = Arc<tokio::sync::RwLock<HashMap<TransportId, LineAccount>>>;

/// Bring up every credential-bound LINE account and run the outbound delivery loops + (optionally)
/// the inbound webhook listener until shutdown. Spawned in-process at host launch.
///
/// `provisioning` is the host's in-process `AccountProvisioning` seam (enumeration + secret
/// resolution stay in-process, no wire crossing). When [`LineConfig::webhook_bind`] is set this
/// binds + serves the adapter-owned webhook listener; otherwise inbound is left for an external
/// ingress to mount [`inbound::webhook_router`] (Phase 2) and this returns once delivery is wired
/// (the detached delivery tasks keep outbound flowing).
pub async fn serve(
    api: Arc<dyn NodeApi>,
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: LineConfig,
    live_clients: LiveClients,
) {
    if !cfg.enabled {
        return;
    }
    let accounts = provisioning.bound_accounts(FAMILY);
    if accounts.is_empty() {
        tracing::info!("line: enabled but no bound line accounts; nothing to do");
        return;
    }

    let ingestor = Arc::new(daemon_ingest::Ingestor::with_policy(
        api.clone(),
        cfg.ingest_policy(),
    ));
    let routes = Arc::new(cfg.routes.clone());

    let mut clients: HashMap<TransportId, LineAccount> = HashMap::new();
    let mut webhook_accounts: HashMap<String, WebhookAccount> = HashMap::new();
    let mut brought_up: Vec<TransportId> = Vec::new();

    for acct in &accounts {
        let Some(blob) = provisioning.account_credential(&acct.credential_ref) else {
            tracing::warn!(
                instance = %acct.transport_instance.as_str(),
                credential_ref = %acct.credential_ref,
                "line: no stored credential for account; run the `line` auth flow first — skipping"
            );
            continue;
        };
        let stored = match StoredCredential::from_blob(&blob) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, credential_ref = %acct.credential_ref, "line: bad credential blob; skipping");
                continue;
            }
        };
        let handle = bare_account(&acct.transport_instance).to_string();
        let account = LineAccount {
            transport: acct.transport_instance.clone(),
            bare: handle.clone(),
            channel_secret: stored.channel_secret.clone(),
            line: line_bot_sdk_rust::client::LINE::new(stored.channel_access_token.clone()),
        };
        clients.insert(acct.transport_instance.clone(), account.clone());
        webhook_accounts.insert(
            handle.clone(),
            WebhookAccount {
                transport: acct.transport_instance.clone(),
                bare: handle,
                channel_secret: stored.channel_secret,
            },
        );
        brought_up.push(acct.transport_instance.clone());
        tracing::info!(instance = %acct.transport_instance.as_str(), "line: account brought up");
    }

    if clients.is_empty() {
        tracing::warn!("line: no accounts could be brought up; exiting");
        return;
    }

    // Publish the live clients so the adapter's `SupportsConversations`/`SupportsContacts` method
    // bodies (which only have `&self`) can resolve the per-account bot client.
    {
        let mut guard = live_clients.write().await;
        for (transport, account) in &clients {
            guard.insert(transport.clone(), account.clone());
        }
    }

    let projector = Arc::new(LineProjector::new(
        api.clone(),
        ingestor.clone(),
        clients.clone(),
    ));
    let delivery = Arc::new(DeliveryManager::new(api.clone(), projector));

    // Resume delivery for any sessions this transport already owns (reconnect / restart), walking the
    // wire pages so every owned session resumes (not just the first page).
    for transport in &brought_up {
        let mut after: Option<String> = None;
        loop {
            let page = api.delivery_sessions(transport.clone(), after.take()).await;
            for session in page.items {
                delivery.ensure(session, transport.clone());
            }
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
    }

    let state = WebhookState {
        ingestor,
        delivery,
        routes,
        accounts: Arc::new(webhook_accounts),
    };

    match &cfg.webhook_bind {
        Some(addr) => match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => {
                tracing::info!(
                    bind = %addr,
                    path = %cfg.webhook_path,
                    "line: serving adapter-owned webhook listener ({}/<handle>)",
                    cfg.webhook_path
                );
                if let Err(e) = serve_webhook(listener, state).await {
                    tracing::warn!(error = %e, "line: webhook listener ended");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, bind = %addr, "line: failed to bind webhook listener; inbound disabled");
            }
        },
        None => {
            tracing::warn!(
                "line: no `webhook_bind` configured — inbound is NOT wired. Phase 2 must mount \
                 `daemon_line::webhook_router` at `{}/<handle>` behind a public URL and register it \
                 as each channel's webhook. Outbound push is active.",
                cfg.webhook_path
            );
        }
    }
}
