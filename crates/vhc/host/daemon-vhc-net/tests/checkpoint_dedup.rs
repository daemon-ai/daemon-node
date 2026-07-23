// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The remote-checkpoint chunk-upload dedup property (design §7.4/§8.3, D-SF3, C5): a live
//! checkpoint's family chunks ride the content-addressed plane, so two consecutive publisher
//! slots whose family is unchanged (bar the chunks that actually moved) upload ONLY the changed
//! chunks + the small document — the amortized remote cost the cadence table depends on, realized
//! by content-addressed `put_content` being idempotent (skip-on-present). This models exactly the
//! `service_op` publisher upload path (`put_content` per referenced family chunk + the document).

use daemon_vhc_net::{ContentStore, MemoryContentStore};

/// Upload a slot's referenced family chunks + document exactly as the publisher seam does.
async fn publish_slot(store: &MemoryContentStore, chunks: &[Vec<u8>], document: &[u8]) {
    for chunk in chunks {
        store.put_content(chunk).await.expect("chunk put");
    }
    store.put_content(document).await.expect("document put");
}

#[tokio::test]
async fn two_publisher_slots_with_an_unchanged_family_upload_only_the_delta() {
    let store = MemoryContentStore::new();

    // A family of three chunks (one publisher slot's `master`), plus its checkpoint document.
    let c0 = b"chunk-0-aaaaaaaa".to_vec();
    let c1 = b"chunk-1-bbbbbbbb".to_vec();
    let c2 = b"chunk-2-cccccccc".to_vec();
    let doc1 = b"checkpoint-document-slot-1".to_vec();
    publish_slot(&store, &[c0.clone(), c1.clone(), c2.clone()], &doc1).await;
    assert_eq!(
        store.object_count(),
        4,
        "slot 1 stores three family chunks + its document"
    );

    // Slot 2: the SAME family except its last chunk moved (c2 -> c2_new); a new document.
    let c2_new = b"chunk-2-dddddddd".to_vec();
    let doc2 = b"checkpoint-document-slot-2".to_vec();
    publish_slot(&store, &[c0.clone(), c1.clone(), c2_new.clone()], &doc2).await;

    // The unchanged chunks (c0, c1) are idempotent no-ops — slot 2 added ONLY the changed chunk
    // and its document. (Whole-family re-upload would have added three more chunks; here: one.)
    assert_eq!(
        store.object_count(),
        6,
        "slot 2 uploads only the changed chunk + its document (c0/c1 dedup to no-ops)"
    );

    // A byte-identical re-publish uploads NOTHING new (the fully-unchanged-family case: zero delta).
    publish_slot(&store, &[c0, c1, c2_new], &doc2).await;
    assert_eq!(
        store.object_count(),
        6,
        "a fully unchanged family + document across a slot uploads nothing"
    );
}
