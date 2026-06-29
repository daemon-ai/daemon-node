// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE PHASE-5 GATE: placement / the first cut (`daemon-workspace-layout.md` §7 phase-5 gate).
//!
//! A child runs in a real isolated OS process — the `daemon` binary in its placed-child role,
//! spawned by `daemon-provision`'s process backend — and is driven across the cut as an ordinary
//! `ManagedUnit`. Its durable state is brokered back to the parent's `SessionStore`, so the parent
//! remains the single fence authority: the happy path completes through the brokered store, and a
//! stale incarnation's commit is fenced across the process boundary (acceptance test #6 across a
//! cut).

use std::sync::Arc;
use std::time::Duration;

use daemon_common::{Budget, ContentHash, JournalStreamId, PartitionId, ReqId, SessionId, UnitId};
use daemon_core::Snapshot;
use daemon_host::PlacedUnit;
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore, TraceSegment};
use daemon_supervision::{
    Ack, EndReason, EventStream, ManageCommand, ManageEvent, ManagedUnit, WorkRef,
};
use daemon_telemetry::{verify_segment, SegmentInput, TraceSigner, GENESIS_ROOT};

const PARTITION: PartitionId = PartitionId::DEFAULT;

/// The node journal signer seed the spawning parent passes down to the placed child (hex of
/// `[0x11; 32]`), so the child's sealed chain verifies under the matching verifying key.
const JOURNAL_SEED_HEX: &str = "1111111111111111111111111111111111111111111111111111111111111111";

/// Spawn the `daemon` binary in its placed-child role as the far side of the cut, passing the node's
/// journal seed so the child journals its durable transcript through the parent's brokered store.
fn placement_spec() -> PlacementSpec {
    PlacementSpec {
        program: env!("CARGO_BIN_EXE_daemon").into(),
        args: Vec::new(),
        env: vec![
            ("DAEMON_PLACED_CHILD".into(), "1".into()),
            ("DAEMON_JOURNAL_SEED".into(), JOURNAL_SEED_HEX.into()),
        ],
    }
}

/// The loaded segment's entries shaped for `verify_segment`.
fn loaded_entries(seg: &TraceSegment) -> Vec<(u64, Vec<u8>, ContentHash)> {
    seg.entries
        .iter()
        .map(|e| (e.seq, e.bytes.clone(), e.content_hash))
        .collect()
}

/// Seed a fresh durable session the placed child will be told to activate.
async fn seed(store: &dyn SessionStore, id: &SessionId) {
    let blob = Snapshot::fresh(id.clone())
        .encode()
        .expect("encode snapshot");
    store
        .create_session(id.clone(), PARTITION, blob)
        .await
        .expect("create session");
}

/// Wait for the placed unit's terminal `ManageEvent` (`Finished` or `Error`) across the cut.
async fn await_terminal(events: &mut EventStream<ManageEvent>) -> ManageEvent {
    loop {
        match tokio::time::timeout(Duration::from_secs(10), events.recv()).await {
            Ok(Ok(ev @ (ManageEvent::Finished { .. } | ManageEvent::Error { .. }))) => return ev,
            Ok(Ok(_)) => continue,
            Ok(Err(_)) => panic!("event stream closed before a terminal event"),
            Err(_) => panic!("timed out waiting for a terminal event across the cut"),
        }
    }
}

/// The happy cut: a child runs in an isolated OS process and completes a turn entirely through the
/// parent's brokered store.
#[tokio::test]
async fn child_runs_in_isolated_process_and_completes_via_brokered_store() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let session = SessionId::new("placed-happy");
    seed(store.as_ref(), &session).await;

    let provisioner = ProcessProvisioner::new();
    let placement = provisioner
        .place(&session, placement_spec())
        .await
        .expect("place child process");
    let unit = PlacedUnit::new(UnitId::new(session.as_str()), placement, store.clone());
    let mut events = unit.events();

    let ack = unit
        .command(ManageCommand::Assign {
            request_id: ReqId(1),
            work: WorkRef::inline("w1", "do the work"),
            budget: Budget::unlimited(),
        })
        .await;
    assert_eq!(ack, Ack::Accepted, "the placed unit should accept the work");

    let terminal = await_terminal(&mut events).await;
    assert!(
        matches!(
            terminal,
            ManageEvent::Finished { ref outcome, .. } if outcome.end_reason == EndReason::Completed
        ),
        "the child should complete across the cut, got {terminal:?}"
    );

    // The brokered parent store — the sole authority — reflects the out-of-process child's commit.
    assert_eq!(
        store.status(&session).await,
        Some(SessionStatus::Completed),
        "the parent store should show the placed child completed"
    );

    unit.command(ManageCommand::Shutdown { drain: false }).await;
}

/// Fencing holds across the cut: a child handed a stale fence cannot commit, because the parent's
/// brokered store rejects it exactly as it would in-process (acceptance test #6 across a process).
#[tokio::test]
async fn fencing_holds_across_the_cut() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let session = SessionId::new("placed-fenced");
    seed(store.as_ref(), &session).await;

    // Ownership transfer: a newer owner takes a higher fence than the one we grant the child.
    let stale = store
        .acquire_activation_lease(&session)
        .await
        .expect("first lease");
    let _current = store
        .acquire_activation_lease(&session)
        .await
        .expect("superseding lease");

    let provisioner = ProcessProvisioner::new();
    let placement = provisioner
        .place(&session, placement_spec())
        .await
        .expect("place child process");
    let unit = PlacedUnit::new(UnitId::new(session.as_str()), placement, store.clone());
    let mut events = unit.events();

    // Drive the child under the STALE fence; its commit must be fenced by the parent store.
    unit.activate_under(session.clone(), stale)
        .await
        .expect("send RunTurn across the cut");

    let terminal = await_terminal(&mut events).await;
    assert!(
        matches!(terminal, ManageEvent::Error { .. }),
        "a stale incarnation must be fenced across the cut, got {terminal:?}"
    );

    // The stale child never committed: the session is not `Completed`.
    assert_ne!(
        store.status(&session).await,
        Some(SessionStatus::Completed),
        "a fenced child must not have committed across the cut"
    );

    unit.command(ManageCommand::Shutdown { drain: false }).await;
}

/// Seat-polish parity: a placed child journals its durable transcript through the parent's brokered
/// store, sealed under the node's seed-derived signer. The parent (sole store authority) ends up
/// holding a sealed segment that verifies under the node's published verifying key — proving the
/// out-of-process child shares the host's journaling seam, not a bespoke unjournaled loop.
#[tokio::test]
async fn placed_child_journals_verifiable_history_via_brokered_store() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let session = SessionId::new("placed-journaled");
    seed(store.as_ref(), &session).await;

    let provisioner = ProcessProvisioner::new();
    let placement = provisioner
        .place(&session, placement_spec())
        .await
        .expect("place child process");
    let unit = PlacedUnit::new(UnitId::new(session.as_str()), placement, store.clone());
    let mut events = unit.events();

    let ack = unit
        .command(ManageCommand::Assign {
            request_id: ReqId(1),
            work: WorkRef::inline("w1", "do the work"),
            budget: Budget::unlimited(),
        })
        .await;
    assert_eq!(ack, Ack::Accepted, "the placed unit should accept the work");

    let terminal = await_terminal(&mut events).await;
    assert!(
        matches!(
            terminal,
            ManageEvent::Finished { ref outcome, .. } if outcome.end_reason == EndReason::Completed
        ),
        "the child should complete across the cut, got {terminal:?}"
    );

    // The durable path keys the journal by the session and seals segment 0 (the first incarnation).
    let stream = JournalStreamId::session(&session);
    let seg = store
        .load_trace_segment(&stream, 0)
        .await
        .expect("the placed child should have journaled a sealed segment into the parent store");
    let committed = seg
        .committed
        .clone()
        .expect("the segment should be sealed (committed root present) after the turn");

    // The chain verifies under the verifying half of the node's seed-derived signer.
    let mut seed_bytes = [0u8; 32];
    for (i, b) in seed_bytes.iter_mut().enumerate() {
        *b = u8::from_str_radix(&JOURNAL_SEED_HEX[i * 2..i * 2 + 2], 16).unwrap();
    }
    let verifying = TraceSigner::from_seed(&seed_bytes).verifying_key();
    let entries = loaded_entries(&seg);
    verify_segment(
        &SegmentInput {
            stream: &stream,
            segment: 0,
            prior: GENESIS_ROOT,
            entries: &entries,
        },
        &committed.root,
        &committed.signature,
        &verifying,
    )
    .expect("the placed child's sealed segment must verify under the node verifying key");

    unit.command(ManageCommand::Shutdown { drain: false }).await;
}
