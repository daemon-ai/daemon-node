//! THE GATE 6a: trace-in-envelope rides the placement cut end-to-end (elfo "context rides every
//! message"), and a placed unit's `Usage` folds into the host's resident metrics dump.
//!
//! A `TraceId` set on the parent *before* driving the out-of-process child is stamped onto the
//! `RunTurn` frame, restored into the child's task-local scope, and stamped back onto every frame
//! the child originates (events, brokered store calls). The parent observes that same id on
//! child-originated frames — proof the context survived a real process boundary and round-tripped.

use std::sync::Arc;
use std::time::Duration;

use daemon_common::{Budget, PartitionId, ReqId, SessionId, TraceId, UnitId};
use daemon_core::Snapshot;
use daemon_host::PlacedUnit;
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore};
use daemon_supervision::{
    Ack, EndReason, EventStream, ManageCommand, ManageEvent, ManagedUnit, WorkRef,
};
use daemon_telemetry::{with_trace, Metrics};

const PARTITION: PartitionId = PartitionId::DEFAULT;

fn placement_spec() -> PlacementSpec {
    PlacementSpec {
        program: env!("CARGO_BIN_EXE_daemon").into(),
        args: Vec::new(),
        // Journal the child too, so its brokered journal appends/seals are also trace-stamped — the
        // trace must ride *every* child-originated frame, including the journaling store calls.
        env: vec![
            ("DAEMON_PLACED_CHILD".into(), "1".into()),
            (
                "DAEMON_JOURNAL_SEED".into(),
                "1111111111111111111111111111111111111111111111111111111111111111".into(),
            ),
        ],
    }
}

async fn seed(store: &dyn SessionStore, id: &SessionId) {
    let blob = Snapshot::fresh(id.clone()).encode().expect("encode snapshot");
    store
        .create_session(id.clone(), PARTITION, blob)
        .await
        .expect("create session");
}

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

/// The parent sets a trace, drives the child, and finds that exact id restored and stamped back on
/// the child's frames — and the child's per-turn `Usage` folded into the resident metrics.
#[tokio::test]
async fn trace_rides_the_cut_and_usage_folds() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let session = SessionId::new("traced-happy");
    seed(store.as_ref(), &session).await;

    let provisioner = ProcessProvisioner::new();
    let placement = provisioner
        .place(&session, placement_spec())
        .await
        .expect("place child process");

    let metrics = Metrics::new();
    let unit = PlacedUnit::with_metrics(
        UnitId::new(session.as_str()),
        placement,
        store.clone(),
        metrics.clone(),
    );
    let mut events = unit.events();

    let trace = TraceId::generate();
    assert!(!trace.is_none());

    // Drive the whole interaction inside the trace scope: the `Assign` (and thus the `RunTurn`
    // frame it sends) is stamped from `current_trace()`.
    let terminal = with_trace(trace, async {
        let ack = unit
            .command(ManageCommand::Assign {
                request_id: ReqId(1),
                work: WorkRef::inline("w1", "do the work"),
                budget: Budget::unlimited(),
            })
            .await;
        assert_eq!(ack, Ack::Accepted);
        await_terminal(&mut events).await
    })
    .await;

    assert!(
        matches!(
            terminal,
            ManageEvent::Finished { ref outcome, .. } if outcome.end_reason == EndReason::Completed
        ),
        "the child should complete across the cut, got {terminal:?}"
    );

    // The proof: the child restored the parent's trace and stamped it back onto its own frames.
    assert_eq!(
        unit.observed_child_trace(),
        trace,
        "the parent-set trace must ride the cut, be restored on the child, and round-trip back"
    );

    // The brokered store still reflects the commit (the trace work did not disturb the cut).
    assert_eq!(store.status(&session).await, Some(SessionStatus::Completed));

    // The child's per-turn usage folded into the host's resident aggregator across the cut.
    let usage = metrics.usage();
    assert!(
        usage.api_calls >= 1,
        "the placed child's Usage event must fold into resident metrics, got {usage:?}"
    );

    unit.command(ManageCommand::Shutdown { drain: false }).await;
}
