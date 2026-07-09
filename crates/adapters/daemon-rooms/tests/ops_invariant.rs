// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! W1-A conformance: the [`assert_ops_match_behavior`] invariant against the real `RoomsAdapter`.
//!
//! The internal loopback transport is store-backed (no network), so it is constructed fully
//! hermetically over an in-memory `SqliteStore` and an ephemeral `TraceSigner`. Its advertised
//! verbs operate on the (empty) store and return `Ok`/`Err(Other)` — never a capability sentinel.

use std::sync::Arc;

use daemon_api::{MessagingProtocol, TransportAdapter};
use daemon_api_testkit::assert_ops_match_behavior;
use daemon_rooms::{RoomsAdapter, RoomsConfig};
use daemon_store::SqliteStore;
use daemon_telemetry::TraceSigner;

#[tokio::test]
async fn ops_match_behavior() {
    let store = Arc::new(SqliteStore::open_in_memory().expect("in-memory store"));
    let signer = Arc::new(TraceSigner::generate());
    let adapter = RoomsAdapter::new(store, signer, RoomsConfig::default());
    let proto: Arc<dyn MessagingProtocol> =
        adapter.messaging().expect("rooms is a messaging protocol");
    assert_ops_match_behavior(proto).await;
}
