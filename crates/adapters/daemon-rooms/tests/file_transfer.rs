// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
// Integration test: raw fs (temp dirs) is expected in tests.
#![allow(clippy::disallowed_methods)]

//! W2-H: the Rooms adapter's loopback [`SupportsFileTransfer`] over the node content store.
//!
//! A Room transfer is a same-node loopback: the file is content-addressed in the node blob store, so
//! `send` verifies the blob resolves and `receive` fetches it. Without a wired blob store the
//! feature is absent (`file_transfer()` → `None`), which keeps the ops-vs-behavior invariant honest.

use std::sync::Arc;

use daemon_api::{FileTransfer, MessagingProtocol, TransportAdapter};
use daemon_api_testkit::assert_ops_match_behavior;
use daemon_host::{BlobStore, FileBlobStore};
use daemon_protocol::TransportId;
use daemon_rooms::{RoomsAdapter, RoomsConfig};
use daemon_store::SqliteStore;
use daemon_telemetry::TraceSigner;

fn temp_root(tag: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "daemon-rooms-ft-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    root
}

async fn adapter_with_blobs(blobs: Arc<dyn BlobStore>) -> Arc<RoomsAdapter> {
    let store = Arc::new(SqliteStore::open_in_memory().expect("in-memory store"));
    let signer = Arc::new(TraceSigner::generate());
    RoomsAdapter::with_blobs(store, signer, RoomsConfig::default(), blobs)
}

#[tokio::test]
async fn file_transfer_send_roundtrips_via_blob_store() {
    let root = temp_root("send");
    let blobs: Arc<dyn BlobStore> = Arc::new(FileBlobStore::open(&root).unwrap());
    let blob = blobs.put(b"the file bytes").await.unwrap();

    let adapter = adapter_with_blobs(blobs).await;
    let ft = adapter
        .clone()
        .messaging()
        .unwrap()
        .file_transfer()
        .expect("blobs wired ⟹ file transfer present");
    assert!(ft.supported().send && ft.supported().receive);

    let transfer = FileTransfer {
        name: "f.bin".into(),
        blob,
        ..Default::default()
    };
    ft.send(TransportId::new("room"), transfer)
        .await
        .expect("loopback send resolves the stored blob");

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn file_transfer_receive_roundtrips_via_blob_store() {
    let root = temp_root("recv");
    let blobs: Arc<dyn BlobStore> = Arc::new(FileBlobStore::open(&root).unwrap());
    let blob = blobs.put(b"inbound bytes").await.unwrap();

    let adapter = adapter_with_blobs(blobs).await;
    let ft = adapter
        .messaging()
        .unwrap()
        .file_transfer()
        .expect("file transfer present");

    let transfer = FileTransfer {
        name: "in.bin".into(),
        blob,
        ..Default::default()
    };
    ft.receive(TransportId::new("room"), transfer)
        .await
        .expect("loopback receive fetches the blob");

    // A missing blob is a (non-sentinel) error, not a silent success.
    let missing = FileTransfer {
        name: "gone.bin".into(),
        blob: daemon_common::BlobRef::new(daemon_common::ContentHash::new([9u8; 32]), 3),
        ..Default::default()
    };
    assert!(ft.receive(TransportId::new("room"), missing).await.is_err());

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn file_transfer_ops_reflect_blob_store() {
    // No blob store ⟹ the feature is absent, and the ops-vs-behavior invariant still holds.
    let store = Arc::new(SqliteStore::open_in_memory().expect("in-memory store"));
    let signer = Arc::new(TraceSigner::generate());
    let bare = RoomsAdapter::new(store, signer, RoomsConfig::default());
    assert!(
        bare.clone().messaging().unwrap().file_transfer().is_none(),
        "no blob store ⟹ file_transfer() is None"
    );
    let proto: Arc<dyn MessagingProtocol> = bare.messaging().unwrap();
    assert_ops_match_behavior(proto).await;

    // With a blob store, the invariant still holds (advertised ⟹ operable).
    let root = temp_root("inv");
    let blobs: Arc<dyn BlobStore> = Arc::new(FileBlobStore::open(&root).unwrap());
    blobs.put(b"x").await.unwrap();
    let adapter = adapter_with_blobs(blobs).await;
    let proto: Arc<dyn MessagingProtocol> = adapter.messaging().unwrap();
    assert_ops_match_behavior(proto).await;
    let _ = std::fs::remove_dir_all(&root);
}
