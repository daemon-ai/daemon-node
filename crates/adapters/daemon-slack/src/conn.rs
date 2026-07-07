// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The in-crate Slack Web API seam — the boundary that keeps `slack-morphism` / `slacko` types out
//! of the adapter's feature-trait method bodies (mirrors the way `daemon-matrix` holds live
//! `matrix_sdk::Client`s behind a registry).
//!
//! [`SlackConn`] is the small async trait the [`crate::adapter::SlackAdapter`] verb bodies call. Two
//! implementations back it: [`MorphismConn`] (bot/app tokens over `slack-morphism`'s hyper client)
//! and [`SlackoConn`] (xoxc/xoxd "stealth" user tokens over `slacko`). Both normalise their SDK's
//! channel list into the SDK-free [`ChannelSummary`] so the adapter never names a Slack type. The
//! trait is object-safe (`Arc<dyn SlackConn>`) so `serve` can register one conn per account and the
//! `&self` verb bodies resolve it from the registry.

use async_trait::async_trait;
use slack_morphism::prelude::*;

use daemon_api::ApiError;

/// An SDK-free projection of a Slack conversation (channel / group / IM) for `list`/`get`/directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSummary {
    /// The channel id (`C…` / `G…` / `D…`).
    pub id: String,
    /// The channel name, when it has one (IMs do not).
    pub name: Option<String>,
    /// The channel topic, when known.
    pub topic: Option<String>,
    /// Whether this is a 1:1 IM (direct message).
    pub is_im: bool,
    /// Whether this is a private channel/group.
    pub is_private: bool,
}

/// The Slack Web API operations the adapter's feature-trait bodies need, normalised to SDK-free
/// types. One live conn per account; resolved from the registry by the `&self` verb bodies.
#[async_trait]
pub trait SlackConn: Send + Sync {
    /// Post `text` to `channel` (`chat.postMessage`).
    async fn post_message(&self, channel: &str, text: &str) -> Result<(), ApiError>;
    /// List the account's visible conversations (`conversations.list`).
    async fn list_channels(&self) -> Result<Vec<ChannelSummary>, ApiError>;
    /// Invite `user` to `channel` (`conversations.invite`).
    async fn invite(&self, channel: &str, user: &str) -> Result<(), ApiError>;
    /// Remove `user` from `channel` (`conversations.kick`).
    async fn kick(&self, channel: &str, user: &str) -> Result<(), ApiError>;
}

/// A bot/app-token conn over `slack-morphism`'s hyper client.
pub struct MorphismConn {
    client: SlackHyperClient,
    token: SlackApiToken,
}

impl MorphismConn {
    /// Build a conn for a bot token (`xoxb-…`). Fails only when the TLS/hyper backend cannot
    /// initialise (a boot-environment defect), surfaced rather than defaulted.
    pub fn new(bot_token: &str) -> Result<Self, ApiError> {
        let connector = SlackClientHyperConnector::new()
            .map_err(|e| ApiError::Other(format!("slack: building hyper connector: {e}")))?;
        let client = SlackClient::new(connector);
        let token = SlackApiToken::new(SlackApiTokenValue(bot_token.to_string()));
        Ok(Self { client, token })
    }
}

#[async_trait]
impl SlackConn for MorphismConn {
    async fn post_message(&self, channel: &str, text: &str) -> Result<(), ApiError> {
        let session = self.client.open_session(&self.token);
        let req = SlackApiChatPostMessageRequest::new(
            SlackChannelId(channel.to_string()),
            SlackMessageContent::new().with_text(text.to_string()),
        );
        session
            .chat_post_message(&req)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("slack chat.postMessage: {e}")))
    }

    async fn list_channels(&self) -> Result<Vec<ChannelSummary>, ApiError> {
        let session = self.client.open_session(&self.token);
        let resp = session
            .conversations_list(&SlackApiConversationsListRequest::new())
            .await
            .map_err(|e| ApiError::Other(format!("slack conversations.list: {e}")))?;
        Ok(resp
            .channels
            .into_iter()
            .map(|c| ChannelSummary {
                id: c.id.0,
                name: c.name,
                // slack-morphism's topic lives behind an optional nested value; the adapter does not
                // surface it on the list projection (the reader can `conversations.info` for detail).
                topic: None,
                is_im: false,
                is_private: false,
            })
            .collect())
    }

    async fn invite(&self, channel: &str, user: &str) -> Result<(), ApiError> {
        let session = self.client.open_session(&self.token);
        let req = SlackApiConversationsInviteRequest::new(
            SlackChannelId(channel.to_string()),
            vec![SlackUserId(user.to_string())],
        );
        session
            .conversations_invite(&req)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("slack conversations.invite: {e}")))
    }

    async fn kick(&self, channel: &str, user: &str) -> Result<(), ApiError> {
        let session = self.client.open_session(&self.token);
        let req = SlackApiConversationsKickRequest::new(
            SlackChannelId(channel.to_string()),
            SlackUserId(user.to_string()),
        );
        session
            .conversations_kick(&req)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("slack conversations.kick: {e}")))
    }
}

/// A user "stealth" conn over `slacko` (xoxc token + xoxd cookie). Gated behind the `stealth` feature
/// (slacko pulls `openssl-sys`, incompatible with the rustls-only build).
#[cfg(feature = "stealth")]
pub struct SlackoConn {
    client: slacko::SlackClient,
}

#[cfg(feature = "stealth")]
impl SlackoConn {
    /// Build a conn for a stealth (xoxc/xoxd) login. Fails only when the inner HTTP client cannot be
    /// built.
    pub fn new(xoxc_token: &str, xoxd_cookie: &str) -> Result<Self, ApiError> {
        let client = slacko::SlackClient::new(slacko::AuthConfig::stealth(xoxc_token, xoxd_cookie))
            .map_err(|e| ApiError::Other(format!("slack: building stealth client: {e}")))?;
        Ok(Self { client })
    }
}

#[cfg(feature = "stealth")]
#[async_trait]
impl SlackConn for SlackoConn {
    async fn post_message(&self, channel: &str, text: &str) -> Result<(), ApiError> {
        self.client
            .chat()
            .post_message(channel, text)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("slack (stealth) chat.postMessage: {e}")))
    }

    async fn list_channels(&self) -> Result<Vec<ChannelSummary>, ApiError> {
        let resp = self
            .client
            .conversations()
            .list()
            .await
            .map_err(|e| ApiError::Other(format!("slack (stealth) conversations.list: {e}")))?;
        Ok(resp
            .channels
            .into_iter()
            .map(|c| ChannelSummary {
                id: c.id,
                name: c.name,
                topic: None,
                is_im: c.is_im.unwrap_or(false),
                is_private: c.is_private.unwrap_or(false),
            })
            .collect())
    }

    async fn invite(&self, channel: &str, user: &str) -> Result<(), ApiError> {
        self.client
            .conversations()
            .invite(channel, &[user])
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("slack (stealth) conversations.invite: {e}")))
    }

    async fn kick(&self, channel: &str, user: &str) -> Result<(), ApiError> {
        self.client
            .conversations()
            .kick(channel, user)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("slack (stealth) conversations.kick: {e}")))
    }
}
