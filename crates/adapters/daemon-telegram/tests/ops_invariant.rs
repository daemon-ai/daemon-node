// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! W1-A conformance: the [`assert_ops_match_behavior`] invariant against the real `TelegramAdapter`,
//! constructed hermetically (no-op provisioning + default config, no live client).

use std::sync::Arc;

use daemon_api::{ApiError, MessagingProtocol, TransportAdapter};
use daemon_api_testkit::assert_ops_match_behavior;
use daemon_host::{AccountProvisioning, ProvisionedAccount};
use daemon_telegram::{TelegramAdapter, TelegramConfig};

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
    let adapter = TelegramAdapter::new(Arc::new(NoProvisioning), TelegramConfig::default());
    let proto: Arc<dyn MessagingProtocol> = adapter
        .messaging()
        .expect("telegram is a messaging protocol");
    assert_ops_match_behavior(proto).await;
}
