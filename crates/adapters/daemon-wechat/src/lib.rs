// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-wechat` — the WeChat (iLink bot) chat transport.
//!
//! An in-process `NodeApi`-client adapter, structurally the same shape as `daemon-matrix`: it isolates
//! the [`wechatbot`] iLink Bot SDK and drives *our* engine as a client. It normalises an iLink
//! long-poll update into an `Origin` and hands it to the reusable `daemon-ingest` gate (inbound);
//! replies flow back through a `daemon-delivery` `Projector` to the peer that opened the session
//! (outbound). No `wechatbot` type leaves this crate.
//!
//! ## Single-mode limitation (iLink account bind)
//!
//! WeChat iLink binds *one* phone account per bot via a QR pairing scan; there is **no user/bot
//! split** and no group/room administration surface. This adapter is therefore **DM-only**: it
//! implements exactly `SupportsConversations` with `send` (honest `supported()`), and its only auth
//! flow is [`AuthFlowKind::QrPairing`](daemon_api::AuthFlowKind::QrPairing). Group chats, membership,
//! roster, directory search, and contact-profile fetch are not part of the iLink bot API and are left
//! unimplemented rather than faked.
//!
//! ## Low-level client (credential authority)
//!
//! The SDK offers a high-level [`WeChatBot`](wechatbot::WeChatBot) dispatcher, but it persists login
//! credentials to its *own* on-disk file and offers no way to inject a stored session. To keep the
//! daemon `CredentialStore` authoritative (spec §6.2) and avoid raw filesystem writes, this adapter
//! drives the low-level [`ILinkClient`](wechatbot::protocol::ILinkClient) directly: the QR login flow
//! (auth), the `getupdates` long-poll (inbound), and `sendmessage` (outbound). The SDK's own internal
//! HTTP (reqwest) is used inside those calls; the adapter issues no direct HTTP itself.
//!
//! ## Ecosystem maturity (user-accepted risk)
//!
//! The WeChat iLink / OpenClaw ecosystem and its Rust crates are young (`wechatbot` is pre-1.0). This
//! is a deliberately-accepted risk: the wire protocol is unofficial and may change, and the crate's
//! API is not yet stable. The adapter keeps all SDK contact behind the thin seams here so a future
//! SDK swap (e.g. to the lower-level `wechat-ilink` crate) stays contained.

#![forbid(unsafe_code)]

pub mod account;
pub mod adapter;
pub mod auth;
pub mod config;
mod inbound;
mod mapping;
mod outbound;

use std::collections::HashMap;
use std::sync::Arc;

use daemon_api::NodeApi;
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

pub use account::{ContextTokens, LiveAccount, WECHAT_QR_BASE_URL};
pub use adapter::WeChatAdapter;
pub use auth::WeChatAuthFlowFactory;
pub use config::WeChatConfig;
pub use inbound::build_reception;
pub use mapping::{bare_account, transport_for, StoredSession};
pub use outbound::{DeliveryManager, WeChatProjector};

use inbound::run_inbound;

/// The transport family this adapter provisions (`AccountProvisioning::bound_accounts`).
pub const FAMILY: &str = "wechat";

/// The shared registry of live, session-restored WeChat accounts keyed by their instance-qualified
/// transport id (`wechat/<user_id>`). Populated by [`serve`] at bring-up and read by the `send` verb
/// body (which only holds `&self`). A `tokio::sync::RwLock` so the read in an `async` verb never
/// blocks the runtime, and bring-up writers don't hold a guard across an await.
pub type LiveClients = Arc<tokio::sync::RwLock<HashMap<TransportId, Arc<LiveAccount>>>>;

/// Bring up every credential-bound WeChat account and run its inbound long-poll + outbound delivery
/// loops until each task is aborted. Spawned in-process at host launch (next to the HTTP surface).
/// `provisioning` is the host's in-process `AccountProvisioning` seam; enumeration + secret
/// resolution stay in-process (no wire crossing).
pub async fn serve(
    api: Arc<dyn NodeApi>,
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: WeChatConfig,
    live: LiveClients,
) {
    if !cfg.enabled {
        return;
    }
    let bound = provisioning.bound_accounts(FAMILY);
    if bound.is_empty() {
        tracing::info!("wechat: enabled but no bound wechat accounts; nothing to do");
        return;
    }

    let ingestor = Arc::new(daemon_ingest::Ingestor::with_policy(
        api.clone(),
        cfg.ingest_policy(),
    ));

    let mut accounts: HashMap<TransportId, Arc<LiveAccount>> = HashMap::new();
    for acct in &bound {
        let Some(blob) = provisioning.account_credential(&acct.credential_ref) else {
            tracing::warn!(
                instance = %acct.transport_instance.as_str(),
                credential_ref = %acct.credential_ref,
                "wechat: no stored session for account; run the wechat QR login first — skipping"
            );
            continue;
        };
        let session = match StoredSession::from_blob(&blob) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, credential_ref = %acct.credential_ref, "wechat: bad session blob; skipping");
                continue;
            }
        };
        let account = LiveAccount::new(session, cfg.bot_agent.as_deref());
        // Announce presence (non-fatal): tells the iLink server this client is online.
        if let Err(e) = account
            .client
            .notify_start(&account.session.base_url, &account.session.token)
            .await
        {
            tracing::warn!(error = %e, instance = %acct.transport_instance.as_str(), "wechat: notify_start failed (ignored)");
        }
        accounts.insert(acct.transport_instance.clone(), account.clone());
        tracing::info!(instance = %acct.transport_instance.as_str(), user = %account.session.user_id, "wechat: account brought up");
    }

    if accounts.is_empty() {
        tracing::warn!("wechat: no accounts could be brought up; exiting");
        return;
    }

    // Publish the live accounts so the adapter's `SupportsConversations::send` body (which only has
    // `&self`) can resolve the per-account client to post through.
    {
        let mut guard = live.write().await;
        for (transport, account) in &accounts {
            guard.insert(transport.clone(), account.clone());
        }
    }

    let projector = Arc::new(WeChatProjector::new(
        api.clone(),
        ingestor.clone(),
        accounts.clone(),
    ));
    let delivery = Arc::new(DeliveryManager::new(api.clone(), projector));

    // Resume delivery for any sessions this transport already owns (reconnect / restart), walking the
    // wire pages so every owned session resumes (not just the first page).
    for transport in accounts.keys() {
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

    // Start each account's inbound long-poll loop.
    let mut poll_tasks = Vec::new();
    for (transport, account) in accounts {
        let ingestor = ingestor.clone();
        let delivery = delivery.clone();
        poll_tasks.push(tokio::spawn(run_inbound(
            account, ingestor, delivery, transport,
        )));
    }

    for task in poll_tasks {
        let _ = task.await;
    }
}
