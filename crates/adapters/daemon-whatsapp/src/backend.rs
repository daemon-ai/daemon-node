// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The backend seam that confines the two WhatsApp SDKs to this crate.
//!
//! The adapter's feature-trait method bodies never touch an SDK type directly: they resolve a
//! [`WaBackend`] from the live registry and call these transport-agnostic verbs. Two implementors:
//! [`crate::backend_bot::BotBackend`] (Meta Cloud API, `wacloudapi`) and
//! [`crate::backend_user::UserBackend`] (WhatsApp Web, `whatsapp-rust`). Each honestly reports the
//! subset it supports via [`WaBackend::membership`] (the Cloud API cannot administer group
//! membership; WhatsApp Web can).

use async_trait::async_trait;

use daemon_api::{ApiError, MembershipOps};

/// A normalised inbound message emitted by a backend's live receive loop (WhatsApp Web only; the
/// Cloud API delivers inbound over Meta webhooks, which need the HTTP surface — Phase 2). Drained by
/// [`crate::serve`] into the reusable `daemon-ingest` gate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaInbound {
    /// The chat id (JID string) the message arrived in — the session/route key + reply address.
    pub chat: String,
    /// The immutable sender identity (JID string).
    pub sender: String,
    /// The message text.
    pub text: String,
    /// Whether the chat is a group (`@g.us`) rather than a 1:1.
    pub is_group: bool,
}

/// The transport-agnostic operations the adapter drives against a live account, per mode. SDK types
/// stay behind the implementors.
#[async_trait]
pub trait WaBackend: Send + Sync {
    /// Send `text` to conversation `to` (a JID for user accounts, a recipient phone/wa-id for bot
    /// accounts). The account itself is always the sender.
    async fn send_text(&self, to: &str, text: &str) -> Result<(), ApiError>;

    /// The membership subset this backend can administer.
    fn membership(&self) -> MembershipOps;

    /// Add `who` (a JID/phone) to group `conv`. Default: unsupported.
    async fn invite(&self, _conv: &str, _who: &str) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_invite".into()))
    }

    /// Remove `who` (a JID/phone) from group `conv`. Default: unsupported.
    async fn remove(&self, _conv: &str, _who: &str) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("member_remove".into()))
    }
}

#[cfg(test)]
pub(crate) mod mock {
    use std::sync::Mutex;

    use super::*;

    /// A record-only backend for the adapter/verb tests (no SDK, no network).
    #[derive(Default)]
    pub struct MockBackend {
        /// `(to, text)` pairs captured by [`WaBackend::send_text`].
        pub sent: Mutex<Vec<(String, String)>>,
        /// Whether this mock advertises + accepts membership ops.
        pub supports_membership: bool,
    }

    #[async_trait]
    impl WaBackend for MockBackend {
        async fn send_text(&self, to: &str, text: &str) -> Result<(), ApiError> {
            self.sent
                .lock()
                .unwrap()
                .push((to.to_string(), text.to_string()));
            Ok(())
        }

        fn membership(&self) -> MembershipOps {
            MembershipOps {
                invite: self.supports_membership,
                remove: self.supports_membership,
                ..MembershipOps::default()
            }
        }

        async fn invite(&self, conv: &str, who: &str) -> Result<(), ApiError> {
            if self.supports_membership {
                self.sent
                    .lock()
                    .unwrap()
                    .push((format!("invite:{conv}"), who.to_string()));
                Ok(())
            } else {
                Err(ApiError::Unsupported("member_invite".into()))
            }
        }

        async fn remove(&self, conv: &str, who: &str) -> Result<(), ApiError> {
            if self.supports_membership {
                self.sent
                    .lock()
                    .unwrap()
                    .push((format!("remove:{conv}"), who.to_string()));
                Ok(())
            } else {
                Err(ApiError::Unsupported("member_remove".into()))
            }
        }
    }
}
