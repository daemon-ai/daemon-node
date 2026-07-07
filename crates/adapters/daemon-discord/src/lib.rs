// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-discord` — the Discord chat transport (Phase 1 of the multi-protocol messaging plan).
//!
//! An in-process `NodeApi`-client adapter, a sibling of [`daemon-matrix`](daemon_matrix): it isolates
//! the `serenity_self` Discord SDK and drives *our* engine as a client. One instance hosts N accounts;
//! each account is a transport instance (`discord/<user_id>`) with its own gateway client and REST
//! handle. The host owns routing: the adapter normalises a gateway message into an `Origin` and hands
//! it to the reusable `daemon-ingest` gate (inbound); replies flow back through a `daemon-delivery`
//! `Projector` to the session's `Primary` channel (outbound). No serenity/Discord type leaves this
//! crate.
//!
//! ## Library: `serenity_self` (dual user + bot tokens)
//!
//! Discord auth is a single opaque token. This adapter uses **`serenity_self`** (a fork of `serenity`
//! ~0.13) because it accepts **both bot and user account tokens** — a superset of `serenity`'s
//! bot-only surface. The account [`DiscordMode`] (`bot` | `user`) selects the interactive-auth flow
//! kind + label. `serenity_self` sends the token verbatim and **panics** on a `Bot `/`Bearer ` prefix,
//! so every token is [`sanitize_token`](account::sanitize_token)'d before it reaches the SDK.
//!
//! ## Warning: user-token (self-bot) mode is a Terms-of-Service / account-ban risk
//!
//! Running against a **user account token** (`mode = "user"`) is self-botting, which violates
//! Discord's Terms of Service and can get the account terminated. This mode exists only because a
//! human operator may knowingly accept that risk for their own account; the default is `bot`.

#![forbid(unsafe_code)]
// serenity's deeply-nested instrumented async gateway futures can approach the default auto-trait
// (`Send`) evaluation recursion limit when spawned; raise it (matches the `daemon-matrix` guidance).
#![recursion_limit = "512"]

pub mod account;
pub mod adapter;
mod auth;
pub mod config;
mod inbound;
mod mapping;
mod outbound;

use std::collections::HashMap;
use std::sync::Arc;

use serenity_self::http::Http;
use serenity_self::model::gateway::GatewayIntents;
use serenity_self::model::id::UserId;
use serenity_self::Client;

use daemon_api::NodeApi;
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

pub use account::{sanitize_token, StoredCredential};
pub use adapter::{DiscordAdapter, LiveClients};
pub use auth::DiscordAuthFlowFactory;
pub use config::{DiscordConfig, DiscordMode, DiscordRoute};
pub use inbound::DiscordHandler;
pub use outbound::{DeliveryManager, DiscordProjector};

use account::bare_account;

/// The transport family this adapter provisions (`AccountProvisioning::bound_accounts`).
pub(crate) const FAMILY: &str = "discord";

/// One brought-up Discord account: its instance-qualified transport id, bare user id, own user id
/// (for self-loop suppression), and the sanitized token (used to build the gateway client).
struct BroughtUp {
    transport: TransportId,
    bare: String,
    me: UserId,
    token: String,
}

/// The gateway intents each account subscribes with. `non_privileged()` covers guild + DM message
/// events; `MESSAGE_CONTENT` (a privileged intent for bots — must be enabled in the developer portal)
/// is added so message bodies arrive. User-account tokens ignore intents (they receive all events).
fn gateway_intents() -> GatewayIntents {
    GatewayIntents::non_privileged() | GatewayIntents::MESSAGE_CONTENT
}

/// Bring up every credential-bound Discord account and run the inbound + outbound loops until each
/// account's gateway ends (or the task is aborted). Spawned in-process at host launch. `provisioning`
/// is the host's in-process `AccountProvisioning` seam; enumeration + token resolution stay
/// in-process (no wire crossing).
pub async fn serve(
    api: Arc<dyn NodeApi>,
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: DiscordConfig,
    live_clients: LiveClients,
) {
    if !cfg.enabled {
        return;
    }
    let accounts = provisioning.bound_accounts(FAMILY);
    if accounts.is_empty() {
        tracing::info!("discord: enabled but no bound discord accounts; nothing to do");
        return;
    }

    let ingestor = Arc::new(daemon_ingest::Ingestor::with_policy(
        api.clone(),
        cfg.ingest_policy(),
    ));
    let routes = Arc::new(cfg.routes.clone());

    // Phase A: resolve each account's identity + REST handle. The registry `Http` (used by the
    // adapter verb bodies + the outbound projector) is built here; the gateway client (Phase C) owns
    // its own `Http`. Two handles per account keeps bring-up free of the handler<->delivery cycle.
    let mut clients: HashMap<TransportId, Arc<Http>> = HashMap::new();
    let mut brought_up: Vec<BroughtUp> = Vec::new();

    for acct in &accounts {
        let Some(blob) = provisioning.account_credential(&acct.credential_ref) else {
            tracing::warn!(
                instance = %acct.transport_instance.as_str(),
                credential_ref = %acct.credential_ref,
                "discord: no stored token for account; run the discord auth flow first — skipping"
            );
            continue;
        };
        let stored = match StoredCredential::from_blob(&blob) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, credential_ref = %acct.credential_ref, "discord: bad credential blob; skipping");
                continue;
            }
        };
        let token = sanitize_token(&stored.token);
        let http = Arc::new(Http::new(&token));
        let me = match http.get_current_user().await {
            Ok(u) => u.id,
            Err(e) => {
                tracing::warn!(error = %e, instance = %acct.transport_instance.as_str(), "discord: token validation failed; skipping");
                continue;
            }
        };

        clients.insert(acct.transport_instance.clone(), http);
        brought_up.push(BroughtUp {
            transport: acct.transport_instance.clone(),
            bare: bare_account(&acct.transport_instance).to_string(),
            me,
            token,
        });
        tracing::info!(instance = %acct.transport_instance.as_str(), user = %me.get(), "discord: account brought up");
    }

    if brought_up.is_empty() {
        tracing::warn!("discord: no accounts could be brought up; exiting");
        return;
    }

    // Publish the live REST handles so the adapter's `Supports*` verb bodies (which only have `&self`)
    // can resolve the per-account handle to execute management verbs against.
    {
        let mut guard = live_clients.write().await;
        for (transport, http) in &clients {
            guard.insert(transport.clone(), http.clone());
        }
    }

    // Phase B: build the outbound projector + delivery manager (needs every account's REST handle).
    let projector = Arc::new(DiscordProjector::new(
        api.clone(),
        ingestor.clone(),
        clients.clone(),
    ));
    let delivery = Arc::new(DeliveryManager::new(api.clone(), projector));

    // Resume delivery for any sessions this transport already owns (reconnect / restart), walking the
    // wire pages so every owned session resumes (not just the first page).
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

    // Phase C: build + start each account's gateway client with its inbound handler.
    let mut gateway_tasks = Vec::new();
    for acct in brought_up {
        let handler = DiscordHandler {
            ingestor: ingestor.clone(),
            delivery: delivery.clone(),
            routes: routes.clone(),
            bare: acct.bare,
            transport: acct.transport.clone(),
            me: acct.me,
        };
        let client = Client::builder(&acct.token, gateway_intents())
            .event_handler(handler)
            .await;
        let mut client = match client {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, instance = %acct.transport.as_str(), "discord: client build failed; skipping");
                continue;
            }
        };
        let instance = acct.transport.clone();
        gateway_tasks.push(tokio::spawn(async move {
            tracing::info!(instance = %instance.as_str(), "discord: gateway loop started");
            if let Err(e) = client.start().await {
                tracing::warn!(error = %e, instance = %instance.as_str(), "discord: gateway loop ended");
            }
        }));
    }

    for task in gateway_tasks {
        let _ = task.await;
    }
}
