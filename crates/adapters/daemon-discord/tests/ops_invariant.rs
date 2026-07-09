// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! W1-A conformance: the [`assert_ops_match_behavior`] invariant against the real `DiscordAdapter`.
//!
//! Constructed hermetically (a no-op provisioning seam + default config, no live client), so every
//! verb the adapter reports unsupported returns the capability sentinel, and advertised verbs
//! short-circuit to a transient `Unsupported("… not connected")` (≠ the sentinel).

use std::sync::Arc;

use daemon_api::{ApiError, MessagingProtocol, TransportAdapter};
use daemon_api_testkit::assert_ops_match_behavior;
use daemon_discord::{DiscordAdapter, DiscordConfig};
use daemon_host::{AccountProvisioning, ProvisionedAccount};

/// A no-op provisioning seam: no bound accounts, no credentials — the adapter comes up unconnected.
struct NoProvisioning;

impl AccountProvisioning for NoProvisioning {
    fn bound_accounts(&self, _family: &str) -> Vec<ProvisionedAccount> {
        Vec::new()
    }
    fn account_credential(&self, _credential_ref: &str) -> Option<String> {
        None
    }
    fn store_account_credential(&self, _credential_ref: &str, _blob: &str) -> Result<(), ApiError> {
        Ok(())
    }
}

#[tokio::test]
async fn ops_match_behavior() {
    let adapter = DiscordAdapter::new(Arc::new(NoProvisioning), DiscordConfig::default());
    let proto: Arc<dyn MessagingProtocol> = adapter
        .messaging()
        .expect("discord is a messaging protocol");
    assert_ops_match_behavior(proto).await;
}
