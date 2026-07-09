// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! W1-A conformance: the [`assert_ops_match_behavior`] invariant against the real `MatrixAdapter`,
//! constructed hermetically (no-op provisioning + default config + no lifecycle sink, no live
//! client). Advertised verbs short-circuit to `Unsupported("matrix account … is not connected")`
//! (≠ the capability sentinel), so the biconditional holds without any homeserver I/O.

use std::sync::Arc;

use daemon_api::{ApiError, MessagingProtocol, TransportAdapter};
use daemon_api_testkit::assert_ops_match_behavior;
use daemon_host::{AccountProvisioning, ProvisionedAccount};
use daemon_matrix::{MatrixAdapter, MatrixConfig};

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
    let adapter = MatrixAdapter::new(Arc::new(NoProvisioning), MatrixConfig::default(), None);
    let proto: Arc<dyn MessagingProtocol> =
        adapter.messaging().expect("matrix is a messaging protocol");
    assert_ops_match_behavior(proto).await;
}
