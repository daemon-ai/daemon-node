// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-slack` — the Slack chat transport (Phase 1 of the multi-protocol messaging plan).
//!
//! An in-process `NodeApi`-client adapter, structurally identical to `daemon-matrix`: it isolates the
//! Slack SDK deps and drives *our* engine as a client. One instance hosts N accounts; each account is
//! a transport instance (`slack/<team-or-label>`) in one of two modes:
//!
//! - **bot/app** — a workspace install (OAuth). Web API + management run through `slack-morphism`;
//!   inbound rides **Socket Mode** (a WSS connection, no public webhook — it needs the app-level
//!   token from [`SlackConfig::app_token`]). One Socket Mode connection serves every workspace the
//!   app is installed in; events are routed to the right account by `team_id`.
//! - **user** — a "stealth" login using a browser-extracted `xoxc` token + `xoxd` cookie, driven by
//!   the [`slacko`](https://crates.io/crates/slacko) crate.
//!
//! Inbound normalises a message event into an `Origin` + `Reception` and hands it to the reusable
//! `daemon-ingest` gate; outbound projects the session's merged log to `chat.postMessage`s via a
//! `daemon-delivery` `Projector`. No `slack-morphism` / `slacko` type leaves this crate (mirrors the
//! matrix/genai/rmcp isolation).
//!
//! # Dependency risk (user-accepted)
//!
//! `slacko` (the xoxc/xoxd "stealth" user-token path) is a young crate (published 2026-01, low
//! download count, single maintainer). It is used **only** for the user-mode conn; the bot/app path
//! (the recommended production mode) rides the mature `slack-morphism`. Stealth tokens are
//! browser-session credentials extracted out-of-band and are inherently more fragile than an OAuth
//! install. Treat the user mode as best-effort.

#![forbid(unsafe_code)]

mod account;
pub mod adapter;
mod auth;
pub mod config;
mod conn;
mod inbound;
mod mapping;
mod outbound;

use std::collections::HashMap;
use std::sync::Arc;

use slack_morphism::prelude::{
    SlackApiToken, SlackApiTokenValue, SlackClient, SlackClientEventsListenerEnvironment,
    SlackClientHyperConnector, SlackClientSocketModeConfig, SlackClientSocketModeListener,
    SlackHyperClient, SlackSocketModeListenerCallbacks,
};

use daemon_api::NodeApi;
use daemon_host::AccountProvisioning;
use daemon_ingest::Ingestor;
use daemon_protocol::TransportId;

pub use account::{bare_account, StoredCredential, FAMILY};
pub use adapter::SlackAdapter;
pub use auth::{SlackBotAuthFlowFactory, SlackUserAuthFlowFactory, USER_FAMILY};
pub use config::{SlackConfig, SlackRoute};
#[cfg(feature = "stealth")]
pub use conn::SlackoConn;
pub use conn::{ChannelSummary, MorphismConn, SlackConn};
pub use inbound::{on_push_event, InboundAccount, InboundState};
pub use outbound::{DeliveryManager, SlackProjector};

use account::bare_account as bare;
use inbound::InboundAccount as Account;

/// The shared registry of live per-account [`SlackConn`]s, keyed by their instance-qualified
/// transport id (`slack/<label>`). Populated by [`serve`] at bring-up and read by the adapter's
/// feature-trait method bodies (which only hold `&self`) to resolve the account's conn. A
/// `tokio::sync::RwLock` so a read in an `async` verb body never blocks the runtime.
pub type LiveConns = Arc<tokio::sync::RwLock<HashMap<TransportId, Arc<dyn SlackConn>>>>;

/// Build the live conn for a user (stealth) credential. Gated on the `stealth` feature: without it
/// (the default, rustls-only build), slacko — and its `openssl-sys` transitive — is not compiled, so
/// user-mode accounts are surfaced as unsupported at bring-up rather than failing the build.
#[cfg(feature = "stealth")]
fn build_user_conn(xoxc: &str, xoxd: &str) -> Result<Arc<dyn SlackConn>, daemon_api::ApiError> {
    Ok(Arc::new(conn::SlackoConn::new(xoxc, xoxd)?))
}

/// The `stealth`-off stub: user-mode conns are unavailable in the default build.
#[cfg(not(feature = "stealth"))]
fn build_user_conn(_xoxc: &str, _xoxd: &str) -> Result<Arc<dyn SlackConn>, daemon_api::ApiError> {
    Err(daemon_api::ApiError::Unsupported(
        "slack user (stealth) mode requires the `stealth` cargo feature (slacko)".into(),
    ))
}

/// Bring up every credential-bound Slack account, wire outbound delivery, and (for bot accounts, when
/// an app-level token is configured) run the Socket Mode inbound loop until aborted. Spawned
/// in-process at host launch. `provisioning` is the host's in-process `AccountProvisioning` seam.
pub async fn serve(
    api: Arc<dyn NodeApi>,
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: SlackConfig,
    live_conns: LiveConns,
) {
    if !cfg.enabled {
        return;
    }
    let accounts = provisioning.bound_accounts(FAMILY);
    if accounts.is_empty() {
        tracing::info!("slack: enabled but no bound slack accounts; nothing to do");
        return;
    }

    let ingestor = Arc::new(Ingestor::with_policy(api.clone(), cfg.ingest_policy()));
    let routes = Arc::new(cfg.routes.clone());

    let mut conns: HashMap<TransportId, Arc<dyn SlackConn>> = HashMap::new();
    let mut transports: Vec<TransportId> = Vec::new();
    // Bot accounts served by Socket Mode, keyed by team id (a single WSS carries all workspaces).
    let mut bot_accounts: HashMap<String, Account> = HashMap::new();

    for acct in &accounts {
        let Some(blob) = provisioning.account_credential(&acct.credential_ref) else {
            tracing::warn!(
                instance = %acct.transport_instance.as_str(),
                credential_ref = %acct.credential_ref,
                "slack: no stored credential for account; run the slack auth flow first — skipping"
            );
            continue;
        };
        let cred = match StoredCredential::from_blob(&blob) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, credential_ref = %acct.credential_ref, "slack: bad credential blob; skipping");
                continue;
            }
        };
        let label = bare(&acct.transport_instance).to_string();
        let conn: Arc<dyn SlackConn> = match &cred {
            StoredCredential::Bot {
                bot_token,
                team_id,
                bot_user_id,
            } => match conn::MorphismConn::new(bot_token) {
                Ok(c) => {
                    bot_accounts.insert(
                        team_id.clone(),
                        Account {
                            bare: label.clone(),
                            transport: acct.transport_instance.clone(),
                            bot_user: bot_user_id.clone(),
                        },
                    );
                    Arc::new(c)
                }
                Err(e) => {
                    tracing::warn!(error = %e, instance = %acct.transport_instance.as_str(), "slack: bot conn build failed; skipping");
                    continue;
                }
            },
            StoredCredential::User {
                xoxc_token,
                xoxd_cookie,
            } => match build_user_conn(xoxc_token, xoxd_cookie) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, instance = %acct.transport_instance.as_str(), "slack: user conn unavailable; skipping");
                    continue;
                }
            },
        };
        conns.insert(acct.transport_instance.clone(), conn);
        transports.push(acct.transport_instance.clone());
        tracing::info!(instance = %acct.transport_instance.as_str(), "slack: account brought up");
    }

    if conns.is_empty() {
        tracing::warn!("slack: no accounts could be brought up; exiting");
        return;
    }

    // Publish the live conns so the adapter's feature-trait bodies (which only have `&self`) can
    // resolve the per-account conn to execute management verbs against.
    {
        let mut guard = live_conns.write().await;
        for (transport, conn) in &conns {
            guard.insert(transport.clone(), conn.clone());
        }
    }

    let projector = Arc::new(outbound::SlackProjector::new(
        api.clone(),
        ingestor.clone(),
        conns.clone(),
    ));
    let delivery = Arc::new(outbound::DeliveryManager::new(api.clone(), projector));

    // Resume delivery for any sessions this transport already owns (reconnect / restart), walking the
    // wire pages so every owned session resumes (not just the first page).
    for transport in &transports {
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

    // Socket Mode inbound: only when there are bot accounts AND an app-level token is configured.
    // User "stealth" accounts never use Socket Mode (that needs an app-level token); their inbound
    // (RTM / polling) is deferred. When inbound is not started, the spawned delivery tasks keep the
    // outbound path alive on their own `Arc`s, so returning here is safe.
    let app_token = match (&cfg.app_token, bot_accounts.is_empty()) {
        (Some(token), false) => token.clone(),
        (None, false) => {
            tracing::info!(
                "slack: bot accounts present but no app-level token configured; Socket Mode inbound disabled"
            );
            return;
        }
        _ => {
            tracing::info!("slack: no bot accounts; Socket Mode inbound disabled");
            return;
        }
    };

    let inbound_state = InboundState {
        ingestor,
        delivery,
        routes,
        accounts: Arc::new(bot_accounts),
    };

    let connector = match SlackClientHyperConnector::new() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "slack: building Socket Mode hyper connector failed; inbound disabled");
            return;
        }
    };
    let client: Arc<SlackHyperClient> = Arc::new(SlackClient::new(connector));
    let environment =
        Arc::new(SlackClientEventsListenerEnvironment::new(client).with_user_state(inbound_state));
    let callbacks = SlackSocketModeListenerCallbacks::new().with_push_events(on_push_event);
    let listener = SlackClientSocketModeListener::new(
        &SlackClientSocketModeConfig::new(),
        environment,
        callbacks,
    );

    let token = SlackApiToken::new(SlackApiTokenValue(app_token));
    if let Err(e) = listener.listen_for(&token).await {
        tracing::warn!(error = %e, "slack: Socket Mode listen_for failed; inbound disabled");
        return;
    }
    listener.start().await;
    tracing::info!("slack: Socket Mode inbound started");

    // Hold the listener (and its WSS clients) alive for the life of the adapter's serve task; the
    // host aborts this future on shutdown.
    let _listener = listener;
    futures::future::pending::<()>().await;
}
