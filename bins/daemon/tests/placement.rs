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

use daemon_common::{Budget, PartitionId, ReqId, SessionId, UnitId};
use daemon_core::Snapshot;
use daemon_host::PlacedUnit;
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore};
use daemon_supervision::{
    Ack, EndReason, EventStream, ManageCommand, ManageEvent, ManagedUnit, WorkRef,
};

const PARTITION: PartitionId = PartitionId::DEFAULT;

/// Spawn the `daemon` binary in its placed-child role as the far side of the cut.
fn placement_spec() -> PlacementSpec {
    PlacementSpec {
        program: env!("CARGO_BIN_EXE_daemon").into(),
        args: Vec::new(),
        env: vec![("DAEMON_PLACED_CHILD".into(), "1".into())],
    }
}

/// Seed a fresh durable session the placed child will be told to activate.
async fn seed(store: &dyn SessionStore, id: &SessionId) {
    let blob = Snapshot::fresh(id.clone()).encode().expect("encode snapshot");
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
