// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Resolved WeChat adapter configuration.
//!
//! WeChat's iLink bot is a **single-mode DM transport** (§ crate header): the bot converses 1:1 with
//! users who message it, with no group-management or room-enumeration surface. There is therefore no
//! route table (as Matrix has) — every inbound DM addresses the agent. The only knobs are whether the
//! adapter is spawned at all and the UA-style `bot_agent` string the SDK stamps on every iLink call.

use daemon_ingest::IngestPolicy;
use daemon_protocol::IsolationPolicy;
use serde::{Deserialize, Serialize};

/// The resolved `[wechat]` config the host hands to [`crate::serve`]. The adapter owns the *shape*;
/// the binary's `NodeConfig` deserializes it (figment) as the `[wechat]` table / `DAEMON_WECHAT__*`
/// env.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WeChatConfig {
    /// Whether the adapter is spawned at all. `false` (default) leaves WeChat off.
    #[serde(with = "daemon_common::flex_bool")]
    pub enabled: bool,
    /// A UA-style identifier of the app driving the bot, stamped as `base_info.bot_agent` on every
    /// iLink API request (e.g. `"daemon/1.0 (prod)"`). Invalid values fall back to the SDK default.
    /// `None` uses the SDK default.
    #[serde(default)]
    pub bot_agent: Option<String>,
}

impl WeChatConfig {
    /// The account-level ingest policy. Isolation is pinned to `PerThread` to match the host's
    /// no-binding routing resolution, so the gate and the outbound stream key busy-state on the same
    /// session id (mirrors `daemon-matrix`).
    pub fn ingest_policy(&self) -> IngestPolicy {
        IngestPolicy {
            isolation: IsolationPolicy::PerThread,
            ..IngestPolicy::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled() {
        let c = WeChatConfig::default();
        assert!(!c.enabled);
        assert!(c.bot_agent.is_none());
    }

    #[test]
    fn ingest_policy_is_per_thread() {
        let c = WeChatConfig::default();
        assert_eq!(c.ingest_policy().isolation, IsolationPolicy::PerThread);
    }
}
