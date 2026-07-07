// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-whatsapp` — the WhatsApp chat transport (Phase 1 of the multi-protocol messaging plan).
//!
//! An in-process `NodeApi`-client adapter modelled on `daemon-matrix`: it normalises inbound WhatsApp
//! messages into `Origin`s and hands them to the reusable `daemon-ingest` gate (inbound), and projects
//! the outbound merged log to WhatsApp messages via a `daemon-delivery` `Projector` (outbound). One
//! instance hosts N accounts; each account is a transport instance (`whatsapp/<handle>`) in one of two
//! modes, selected by the stored credential:
//!
//! * **user** (`whatsapp-rust`) — a WhatsApp Web linked device. Login is QR pairing; the paired
//!   `Device` snapshot is the persisted session blob. Inbound is event-bus driven; outbound send +
//!   group participant add/remove are wired.
//! * **bot** (`wacloudapi`) — a Meta Cloud API business number. Outbound send is wired; inbound
//!   arrives over Meta webhooks, which need the HTTP surface (Phase 2), so no inbound is wired here.
//!
//! All SDK types stay confined to this crate behind the [`backend::WaBackend`] seam (mirrors the
//! matrix-sdk / genai / rmcp isolation elsewhere in the tree).
//!
//! # Disclaimer
//!
//! `whatsapp-rust` is an **unofficial** reimplementation of the WhatsApp Web protocol. Running a
//! custom WhatsApp client may violate Meta's Terms of Service and can result in the account being
//! banned. The user accepts this risk when they pair a `mode = "user"` account. The `mode = "bot"`
//! backend uses Meta's official Cloud API and carries no such risk.

#![forbid(unsafe_code)]
// whatsapp-rust's deeply-nested instrumented async futures can overflow the default auto-trait
// (`Send`) evaluation recursion limit when spawned; raise it (same guard the Matrix adapter uses).
#![recursion_limit = "512"]

mod account;
pub mod adapter;
pub mod auth;
pub mod backend;
mod backend_bot;
mod backend_user;
pub mod config;
mod inbound;
mod mapping;
mod outbound;

use std::collections::HashMap;
use std::sync::Arc;

use daemon_api::NodeApi;
use daemon_host::{with_request_context, AccountProvisioning, RequestContext};
use daemon_protocol::TransportId;

use whatsapp_rust::bot::BotHandle;

use crate::account::{bare_account, StoredCredential, FAMILY};
use crate::backend::WaBackend;
use crate::backend_bot::BotBackend;
use crate::backend_user::UserBackend;
use crate::inbound::{handle_inbound, InboundCtx};
use crate::outbound::{DeliveryManager, WhatsappProjector};

pub use account::StoredCredential as WhatsappCredential;
pub use adapter::WhatsappAdapter;
pub use auth::WhatsappAuthFlowFactory;
pub use backend::{WaBackend as WhatsappBackend, WaInbound};
pub use config::{WhatsappConfig, WhatsappRoute};

/// The family constant this adapter provisions (`AccountProvisioning::bound_accounts`).
pub use account::FAMILY as WHATSAPP_FAMILY;

/// The shared registry of live per-account backends keyed by their instance-qualified transport id
/// (`whatsapp/<handle>`). Populated by [`serve`] at bring-up and read by the adapter's
/// `SupportsConversations`/`SupportsMembership` method bodies (which only hold `&self`). A
/// `tokio::sync::RwLock` so the read in an `async` verb body never blocks the runtime.
pub type LiveBackends = Arc<tokio::sync::RwLock<HashMap<TransportId, Arc<dyn WaBackend>>>>;

/// The inbound channel depth per user account (bounded backpressure onto the SDK event bus).
const INBOUND_CHANNEL_DEPTH: usize = 64;

/// Bring up every credential-bound WhatsApp account and run the inbound + outbound loops. Spawned
/// in-process at host launch. `provisioning` is the host's in-process `AccountProvisioning` seam; the
/// live backends are published into `backends` so the adapter's verb bodies can send through them.
pub async fn serve(
    api: Arc<dyn NodeApi>,
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: WhatsappConfig,
    backends: LiveBackends,
) {
    if !cfg.enabled {
        return;
    }
    let accounts = provisioning.bound_accounts(FAMILY);
    if accounts.is_empty() {
        tracing::info!("whatsapp: enabled but no bound whatsapp accounts; nothing to do");
        return;
    }

    let ingestor = Arc::new(daemon_ingest::Ingestor::with_policy(
        api.clone(),
        cfg.ingest_policy(),
    ));
    let routes = Arc::new(cfg.routes.clone());
    let projector = Arc::new(WhatsappProjector::new(
        api.clone(),
        ingestor.clone(),
        backends.clone(),
    ));
    let delivery = Arc::new(DeliveryManager::new(api.clone(), projector));

    let mut handles: Vec<BotHandle> = Vec::new();
    let mut brought_up: Vec<TransportId> = Vec::new();

    for acct in &accounts {
        let transport = acct.transport_instance.clone();
        let Some(blob) = provisioning.account_credential(&acct.credential_ref) else {
            tracing::warn!(
                instance = %transport.as_str(),
                credential_ref = %acct.credential_ref,
                "whatsapp: no stored credential for account; run `daemon whatsapp login` first — skipping"
            );
            continue;
        };
        let stored = match StoredCredential::from_blob(&blob) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, credential_ref = %acct.credential_ref, "whatsapp: bad credential blob; skipping");
                continue;
            }
        };

        let backend: Arc<dyn WaBackend> = match stored {
            StoredCredential::Bot {
                access_token,
                phone_number_id,
            } => Arc::new(BotBackend::new(&access_token, &phone_number_id)),
            StoredCredential::User { device, .. } => {
                let (tx, mut rx) = tokio::sync::mpsc::channel(INBOUND_CHANNEL_DEPTH);
                match UserBackend::connect(device, tx).await {
                    Ok((be, handle)) => {
                        handles.push(handle);
                        // Drain this account's inbound stream into the reusable ingest gate.
                        let ctx = InboundCtx {
                            ingestor: ingestor.clone(),
                            delivery: delivery.clone(),
                            routes: routes.clone(),
                            account: bare_account(&transport).to_string(),
                            transport: transport.clone(),
                        };
                        tokio::spawn(async move {
                            while let Some(inbound) = rx.recv().await {
                                handle_inbound(&ctx, inbound).await;
                            }
                        });
                        be
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, instance = %transport.as_str(), "whatsapp: user account connect failed; skipping");
                        continue;
                    }
                }
            }
        };

        backends.write().await.insert(transport.clone(), backend);
        brought_up.push(transport.clone());
        tracing::info!(instance = %transport.as_str(), "whatsapp: account brought up");
    }

    if brought_up.is_empty() {
        tracing::warn!("whatsapp: no accounts could be brought up; exiting");
        return;
    }

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

    // Park on the user-mode run loops (bot-only deployments have none — the spawned delivery/inbound
    // tasks keep the projector + registry alive via their `Arc`s, and the adapter holds `backends`).
    // Bind the in-process `internal` principal so any awaited work inherits the trusted identity.
    with_request_context(RequestContext::internal(), async move {
        for handle in handles {
            let _ = handle.await;
        }
    })
    .await;
}
