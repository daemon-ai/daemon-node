// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Per-account runtime state: the live iLink client, the restored session, and the per-peer reply
//! context-token map.
//!
//! Each WeChat **account** is a distinct transport instance (`wechat/<user_id>`) owning one
//! [`ILinkClient`] and one long-poll loop (spec §2, mirroring `daemon-matrix`). WeChat's `sendmessage`
//! requires a per-peer *context token* (an opaque reply handle the server hands out on each inbound
//! message); we remember the newest token per peer in [`ContextTokens`] so an outbound reply — which
//! runs after the inbound message that carried the token — can address the right conversation. This
//! mirrors the SDK's own internal `context_tokens` map, lifted out so the outbound projector (which
//! only holds `&self`) can read it.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use wechatbot::protocol::ILinkClient;

use crate::mapping::StoredSession;

/// The QR-login host WeChat iLink always issues login QR codes from (the SDK's `FIXED_QR_BASE_URL`).
/// Polling may later be redirected to an IDC host; the *minted* session's `base_url` is what
/// [`StoredSession`] persists and bring-up dials.
pub const WECHAT_QR_BASE_URL: &str = "https://ilinkai.weixin.qq.com";

/// The newest reply context-token per peer user id (`user_id -> context_token`). Shared (a
/// `tokio::sync::RwLock` so an `async` verb/projector body never blocks the runtime) between the
/// inbound loop (writer) and the outbound projector + `conv_send` verb (readers).
pub type ContextTokens = Arc<RwLock<HashMap<String, String>>>;

/// One brought-up WeChat account: the live iLink client, its restored session (token + base URL +
/// identity), and the per-peer context-token map. Held behind an `Arc` in the adapter's live-account
/// registry so the `&self` verb bodies and the outbound projector can resolve it.
pub struct LiveAccount {
    /// The low-level iLink API client (shared; its internal HTTP client is `Send + Sync`).
    pub client: Arc<ILinkClient>,
    /// The restored session material (token + poll base URL + identity).
    pub session: StoredSession,
    /// The newest reply context-token per peer.
    pub context_tokens: ContextTokens,
}

impl LiveAccount {
    /// Build a live account over a fresh iLink client for `session`, stamped with the optional
    /// `bot_agent` (invalid values fall back to the SDK default inside `with_bot_agent`).
    pub fn new(session: StoredSession, bot_agent: Option<&str>) -> Arc<Self> {
        Arc::new(Self {
            client: Arc::new(ILinkClient::with_bot_agent(bot_agent)),
            session,
            context_tokens: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Remember the newest reply `context_token` for peer `user_id` (called on each inbound message).
    pub async fn remember_context(&self, user_id: &str, context_token: &str) {
        if user_id.is_empty() || context_token.is_empty() {
            return;
        }
        self.context_tokens
            .write()
            .await
            .insert(user_id.to_string(), context_token.to_string());
    }

    /// The most recent reply context-token for peer `user_id`, if one has been seen.
    pub async fn context_for(&self, user_id: &str) -> Option<String> {
        self.context_tokens.read().await.get(user_id).cloned()
    }
}
