// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE PHASE-6b GATE: a verifiable, signed, hash-linked trace journal bound to the durable
//! incarnation (`daemon-workspace-layout.md` §7 phase-6: "trace-in-envelope"). A unit's
//! `ManageEvent`s are journaled as Gordian Envelopes, the per-`(session, epoch)` Merkle root is
//! signed and committed under the checkpoint fence, and verification recomputes the root from
//! the durable bytes — detecting tampering, proving the cross-epoch chain, and rejecting a stale
//! incarnation's seal.

use daemon_common::{ContentHash, JournalStreamId, MerkleRoot, SessionId};
use daemon_common::{PartitionId, SnapshotBlob, UsageDelta};
use daemon_host::JournalSink;
use daemon_store::{InMemoryStore, SessionStore, StoreError, TraceSegment};
use daemon_supervision::{EndReason, ManageEvent, Outcome, StartTrigger};
use daemon_telemetry::{verify_segment, SegmentInput, TraceSigner, VerifyError, GENESIS_ROOT};
use std::sync::Arc;

const PARTITION: PartitionId = PartitionId::DEFAULT;

fn turn_events() -> Vec<ManageEvent> {
    vec![
        ManageEvent::Started {
            seq: 0,
            trigger: StartTrigger::Resumed,
        },
        ManageEvent::Usage {
            seq: 1,
            delta: UsageDelta {
                input_tokens: 12,
                output_tokens: 8,
                api_calls: 1,
                ..Default::default()
            },
        },
        ManageEvent::Finished {
            seq: 2,
            outcome: Outcome::ended(EndReason::Completed),
        },
    ]
}

async fn seed(store: &InMemoryStore, id: &SessionId) {
    store
        .create_session(id.clone(), PARTITION, SnapshotBlob::default())
        .await
        .unwrap();
}

fn input_from<'a>(
    stream: &'a JournalStreamId,
    segment: u64,
    prior: MerkleRoot,
    entries: &'a [(u64, Vec<u8>, ContentHash)],
) -> SegmentInput<'a> {
    SegmentInput {
        stream,
        segment,
        prior,
        entries,
    }
}

fn loaded_entries(seg: &TraceSegment) -> Vec<(u64, Vec<u8>, ContentHash)> {
    seg.entries
        .iter()
        .map(|e| (e.seq, e.bytes.clone(), e.content_hash))
        .collect()
}

/// Record a turn's events, seal the signed root under the fence, and verify it from the store.
/// Then prove tampering with any persisted entry is detected.
#[tokio::test]
async fn record_commit_verify_and_tamper_detect() {
    let store = Arc::new(InMemoryStore::new());
    let id = SessionId::new("journal-happy");
    seed(&store, &id).await;
    let fence = store.acquire_activation_lease(&id).await.unwrap();

    let signer = Arc::new(TraceSigner::generate());
    let stream = JournalStreamId::session(&id);
    let sink = JournalSink::for_incarnation(
        store.clone() as Arc<dyn SessionStore>,
        signer.clone(),
        stream.clone(),
        fence,
        0,
    )
    .await;

    for ev in turn_events() {
        sink.record(&ev).await.unwrap();
    }
    let root = sink.seal().await.unwrap();

    // Load the sealed segment from the store and verify it end to end.
    let seg = store.load_trace_segment(&stream, 0).await.unwrap();
    let committed = seg.committed.clone().expect("segment sealed");
    assert_eq!(committed.root, root);
    let entries = loaded_entries(&seg);
    verify_segment(
        &input_from(&stream, 0, GENESIS_ROOT, &entries),
        &committed.root,
        &committed.signature,
        &signer.verifying_key(),
    )
    .expect("a faithfully sealed segment verifies from the store");

    // Tamper: mutate a persisted entry's bytes; verification must fail.
    let mut tampered = entries.clone();
    tampered[1].1[0] ^= 0xFF;
    let err = verify_segment(
        &input_from(&stream, 0, GENESIS_ROOT, &tampered),
        &committed.root,
        &committed.signature,
        &signer.verifying_key(),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            VerifyError::Decode | VerifyError::ContentHashMismatch | VerifyError::RootMismatch
        ),
        "tampering must be detected, got {err:?}"
    );
}

/// Epoch N+1's root chains onto epoch N's: a sink opened `chained` onto the prior root produces
/// a root that only verifies with that prior (a broken link is a root mismatch).
#[tokio::test]
async fn cross_epoch_chain() {
    let store = Arc::new(InMemoryStore::new());
    let id = SessionId::new("journal-chain");
    seed(&store, &id).await;

    // Epoch 0.
    let f0 = store.acquire_activation_lease(&id).await.unwrap();
    let signer = Arc::new(TraceSigner::generate());
    let stream = JournalStreamId::session(&id);
    let sink0 = JournalSink::for_incarnation(
        store.clone() as Arc<dyn SessionStore>,
        signer.clone(),
        stream.clone(),
        f0,
        0,
    )
    .await;
    for ev in turn_events() {
        sink0.record(&ev).await.unwrap();
    }
    let root0 = sink0.seal().await.unwrap();

    // Epoch 1 chains onto epoch 0's sealed root (loaded from the store), under a fresh
    // (superseding) fence.
    let f1 = store.acquire_activation_lease(&id).await.unwrap();
    let sink1 = JournalSink::for_incarnation(
        store.clone() as Arc<dyn SessionStore>,
        signer.clone(),
        stream.clone(),
        f1,
        1,
    )
    .await;
    for ev in turn_events() {
        sink1.record(&ev).await.unwrap();
    }
    let root1 = sink1.seal().await.unwrap();

    let seg1 = store.load_trace_segment(&stream, 1).await.unwrap();
    let committed1 = seg1.committed.clone().unwrap();
    let entries1 = loaded_entries(&seg1);

    // Verifies with the true prior (root0)...
    verify_segment(
        &input_from(&stream, 1, root0, &entries1),
        &committed1.root,
        &committed1.signature,
        &signer.verifying_key(),
    )
    .expect("the chained segment verifies with the correct prior");
    assert_eq!(committed1.root, root1);

    // ...and a broken link (wrong prior) is rejected.
    assert_eq!(
        verify_segment(
            &input_from(&stream, 1, GENESIS_ROOT, &entries1),
            &committed1.root,
            &committed1.signature,
            &signer.verifying_key(),
        )
        .unwrap_err(),
        VerifyError::RootMismatch
    );
}

/// A stale incarnation cannot seal a segment root: the commit is fenced exactly as a checkpoint
/// would be (reuses acceptance tests #4/#6, now for the trace root).
#[tokio::test]
async fn stale_fence_cannot_seal() {
    let store = Arc::new(InMemoryStore::new());
    let id = SessionId::new("journal-fenced");
    seed(&store, &id).await;

    let stale = store.acquire_activation_lease(&id).await.unwrap();
    // Ownership transfers to a newer incarnation.
    let _current = store.acquire_activation_lease(&id).await.unwrap();

    let signer = Arc::new(TraceSigner::generate());
    let stream = JournalStreamId::session(&id);
    let sink = JournalSink::for_incarnation(
        store.clone() as Arc<dyn SessionStore>,
        signer,
        stream.clone(),
        stale,
        0,
    )
    .await;
    for ev in turn_events() {
        // Appends are not fenced (the open log), but the seal is.
        sink.record(&ev).await.unwrap();
    }
    let r = sink.seal().await;
    assert!(
        matches!(r, Err(StoreError::Fenced { .. })),
        "a stale incarnation must not seal a segment root, got {r:?}"
    );
    // No root was committed.
    let seg = store.load_trace_segment(&stream, 0).await.unwrap();
    assert!(seg.committed.is_none());
}
