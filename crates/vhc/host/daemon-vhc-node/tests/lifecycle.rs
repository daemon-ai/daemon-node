// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The durable run-instance state machine end-to-end over the `WorkerControl` seam: terminal
//! transitions with observed-teardown release ordering, idempotent duplicate terminals,
//! stale-generation discard, the bounded retry budget with escalation, reconvergence as a fresh
//! incarnation, crash-window repair on startup, and transport-loss observation via pump-stream
//! closure. The worker is a scripted streaming fake — the real-subprocess twin lives in the
//! supervisor suite.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{VhcApi, VhcPolicy, VhcPolicyMode};
use daemon_vhc_node::service::{VhcError, WorkerControl};
use daemon_vhc_node::{RunState, VhcService, VhcServiceParts, VhcStore};
use daemon_vhc_session::config::{RetryConfig, VhcConfig};
use daemon_vhc_session::protocol::{
    self, Eligibility, Hardware, JoinPolicy, LeaveMode, TerminalOutcome,
};
use tokio::sync::mpsc::UnboundedSender;

/// A scripted streaming worker: every `join_streaming` hands back a LIVE receiver whose sender
/// the test holds, so the pump stream stays open until the test terminates or severs it.
#[derive(Default)]
struct StreamingWorker {
    joins: Mutex<Vec<String>>,
    leaves: Mutex<Vec<String>>,
    streams: Mutex<Vec<UnboundedSender<protocol::Event>>>,
    throttles: Mutex<Vec<(Option<u32>, Option<u8>, bool)>>,
}

impl StreamingWorker {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    fn joins(&self) -> Vec<String> {
        self.joins.lock().unwrap().clone()
    }
    /// Sever every live stream (drop the senders) — the transport-loss simulation.
    fn sever_streams(&self) {
        self.streams.lock().unwrap().clear();
    }
}

#[async_trait]
impl WorkerControl for StreamingWorker {
    async fn probe(&self) -> Result<Hardware, VhcError> {
        Ok(Hardware {
            gpus: 1,
            vram_mb: 24_000,
            ram_mb: 64_000,
            backend_lanes: vec!["cpu".into()],
            ..Default::default()
        })
    }
    async fn assess(
        &self,
        _envelope: Vec<u8>,
        _role: Option<String>,
    ) -> Result<Eligibility, VhcError> {
        Ok(Eligibility {
            eligible: true,
            ..Default::default()
        })
    }
    async fn join(
        &self,
        run_id: String,
        _coordinator: String,
        _credentials: Vec<u8>,
        _policy: JoinPolicy,
        _admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<(), VhcError> {
        self.joins.lock().unwrap().push(run_id);
        Ok(())
    }
    async fn join_streaming(
        &self,
        run_id: String,
        _coordinator: String,
        _credentials: Vec<u8>,
        _policy: JoinPolicy,
        _admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<protocol::Event>, VhcError> {
        self.joins.lock().unwrap().push(run_id);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.streams.lock().unwrap().push(tx);
        Ok(rx)
    }
    async fn leave(&self, run_id: String, _mode: LeaveMode) -> Result<(), VhcError> {
        self.leaves.lock().unwrap().push(run_id);
        Ok(())
    }
    async fn throttle(
        &self,
        vram_cap_mb: Option<u32>,
        duty_cycle_pct: Option<u8>,
        paused: bool,
    ) -> Result<(), VhcError> {
        self.throttles
            .lock()
            .unwrap()
            .push((vram_cap_mb, duty_cycle_pct, paused));
        Ok(())
    }
}

fn config(max_retries: u32) -> VhcConfig {
    VhcConfig {
        enabled: true,
        retry: RetryConfig {
            max_retries,
            initial_backoff_ms: 1, // effectively-immediate schedules for deterministic ticks
            max_backoff_ms: 2,
            min_uptime_ms: 60_000,
            reconcile_tick_ms: 1_000,
        },
        ..VhcConfig::default()
    }
}

fn policy() -> VhcPolicy {
    VhcPolicy {
        mode: VhcPolicyMode::Idle,
        vram_cap_mb: 8_000,
        duty_cycle_pct: 90,
        schedule: None,
    }
}

fn service_over(
    store: VhcStore,
    worker: Arc<StreamingWorker>,
    max_retries: u32,
) -> Arc<VhcService> {
    let svc = Arc::new(VhcService::new(VhcServiceParts {
        config: config(max_retries),
        store,
        worker,
        feed: None,
        discovery: None,
        budget: None,
        worker_factory: None,
        identity_dir: None,
    }));
    svc.bind_self();
    svc
}

fn terminated(run: &str, generation: u64, outcome: TerminalOutcome) -> protocol::Event {
    protocol::Event::RunTerminated {
        run_id: run.to_string(),
        generation,
        outcome,
    }
}

/// Poll the store until `pred` holds (bounded) — the pump/tick paths are asynchronous.
async fn wait_for(svc: &VhcService, run: &str, pred: impl Fn(RunState) -> bool) {
    for _ in 0..200 {
        if let Ok(Some(row)) = svc.store().get_run(run) {
            if pred(row.run_state) {
                return;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let state = svc.store().get_run(run).unwrap().map(|r| r.run_state);
    panic!("run `{run}` never reached the expected state (last: {state:?})");
}

/// A module run end transitions `completed`, releases the ledger reservation only after the
/// durable observation, removes the live instance — and the run is NEVER rejoined, neither by a
/// restart pass nor by the reconciliation tick.
#[tokio::test]
async fn completed_releases_ledger_and_never_rejoins() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vhc.db");
    {
        let worker = StreamingWorker::new();
        let svc = service_over(VhcStore::open(&path).unwrap(), worker.clone(), 5);
        svc.vhc_join("run-c".into(), policy(), "op".into())
            .await
            .unwrap();
        assert_eq!(svc.arbiter().instances(), 1, "the join reserved the ledger");
        let generation = svc.store().get_run("run-c").unwrap().unwrap().instance;

        let outs = svc
            .handle_worker_event(&terminated(
                "run-c",
                generation,
                TerminalOutcome::Completed { outcome: 0 },
            ))
            .unwrap();
        assert_eq!(outs.len(), 1, "the transition surfaces one app event");
        assert_eq!(svc.arbiter().instances(), 0, "observed teardown released");
        let row = svc.store().get_run("run-c").unwrap().unwrap();
        assert_eq!(row.run_state, RunState::Completed);
        assert_eq!(row.pending_run_state, None, "the release committed");
        assert!(row
            .terminal_reason
            .as_deref()
            .is_some_and(|r| r.contains("run end")));
        // The tick never resurrects a completed run.
        assert_eq!(svc.reconcile_tick().await.unwrap(), 0);
        assert_eq!(worker.joins().len(), 1);
    }
    // A restart over the same vhc.db reconverges NOTHING for the completed run.
    let worker = StreamingWorker::new();
    let svc = service_over(VhcStore::open(&path).unwrap(), worker.clone(), 5);
    assert_eq!(svc.start().await.unwrap(), 0, "completed never rejoins");
    assert!(worker.joins().is_empty());
}

/// Duplicate terminal delivery is idempotent: the second event transitions nothing and cannot
/// double-release the ledger (a fresh admission after the first release stays intact).
#[tokio::test]
async fn duplicate_terminal_is_idempotent() {
    let worker = StreamingWorker::new();
    let svc = service_over(VhcStore::open_in_memory().unwrap(), worker.clone(), 5);
    svc.vhc_join("run-d".into(), policy(), "op".into())
        .await
        .unwrap();
    let generation = svc.store().get_run("run-d").unwrap().unwrap().instance;
    let ev = terminated(
        "run-d",
        generation,
        TerminalOutcome::Completed { outcome: 0 },
    );

    assert_eq!(svc.handle_worker_event(&ev).unwrap().len(), 1);
    assert_eq!(svc.arbiter().instances(), 0);
    // Another run's reservation stands in for "state a double-release could corrupt".
    svc.vhc_join("run-other".into(), policy(), "op2".into())
        .await
        .unwrap();
    assert_eq!(svc.arbiter().instances(), 1);

    let outs = svc.handle_worker_event(&ev).unwrap();
    assert!(outs.is_empty(), "the duplicate transitions nothing");
    assert_eq!(svc.arbiter().instances(), 1, "no double release");
    assert_eq!(
        svc.store().get_run("run-d").unwrap().unwrap().run_state,
        RunState::Completed
    );
}

/// Events stamped with a stale generation are discarded whole: they fold no contribution,
/// transition no state, and can never terminate the CURRENT instance.
#[tokio::test]
async fn stale_generation_events_are_discarded() {
    let worker = StreamingWorker::new();
    let svc = service_over(VhcStore::open_in_memory().unwrap(), worker.clone(), 5);
    svc.vhc_join("run-g".into(), policy(), "op".into())
        .await
        .unwrap();
    let generation = svc.store().get_run("run-g").unwrap().unwrap().instance;
    let stale = generation + 100;

    // A stale RunPhase neither surfaces nor persists.
    let outs = svc
        .handle_worker_event(&protocol::Event::RunPhase {
            run_id: "run-g".into(),
            phase: "hijack".into(),
            epoch: 9,
            round: 99,
            generation: stale,
        })
        .unwrap();
    assert!(outs.is_empty());
    let row = svc.store().get_run("run-g").unwrap().unwrap();
    assert_ne!(row.last_phase, "hijack");
    assert_ne!(row.last_round, 99);

    // A stale terminal cannot end the current instance or release its reservation.
    let outs = svc
        .handle_worker_event(&terminated(
            "run-g",
            stale,
            TerminalOutcome::FailedTerminal {
                reason: "stale".into(),
            },
        ))
        .unwrap();
    assert!(outs.is_empty());
    assert_eq!(svc.arbiter().instances(), 1, "the reservation stands");
    assert_eq!(
        svc.store().get_run("run-g").unwrap().unwrap().run_state,
        RunState::Running
    );

    // The current generation's terminal still lands (the gate is per-generation, not a latch).
    svc.handle_worker_event(&terminated(
        "run-g",
        generation,
        TerminalOutcome::Completed { outcome: 0 },
    ))
    .unwrap();
    assert_eq!(
        svc.store().get_run("run-g").unwrap().unwrap().run_state,
        RunState::Completed
    );
}

/// A recoverable failure consumes budget and reconverges as a FRESH incarnation via the
/// reconciliation tick; exhaustion escalates to a typed terminal failure that never reconverges.
#[tokio::test]
async fn retry_budget_reconverges_then_escalates_on_exhaustion() {
    let worker = StreamingWorker::new();
    let svc = service_over(VhcStore::open_in_memory().unwrap(), worker.clone(), 1);
    svc.vhc_join("run-r".into(), policy(), "op".into())
        .await
        .unwrap();
    let first_gen = svc.store().get_run("run-r").unwrap().unwrap().instance;

    // First recoverable failure: within budget → failed_retryable, one attempt consumed.
    svc.handle_worker_event(&terminated(
        "run-r",
        first_gen,
        TerminalOutcome::FailedRetryable {
            reason: "transport loss".into(),
        },
    ))
    .unwrap();
    let row = svc.store().get_run("run-r").unwrap().unwrap();
    assert_eq!(row.run_state, RunState::FailedRetryable);
    assert_eq!(row.retry_count, 1);
    assert!(row.next_retry_ms.is_some(), "a reconvergence is scheduled");
    assert_eq!(svc.arbiter().instances(), 0, "the failed instance released");

    // The tick fires the due reconvergence: a NEW incarnation joins (generation advances).
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    assert_eq!(svc.reconcile_tick().await.unwrap(), 1);
    let row = svc.store().get_run("run-r").unwrap().unwrap();
    assert_eq!(row.run_state, RunState::Running);
    let second_gen = row.instance;
    assert!(second_gen > first_gen, "mid-run reconvergence mints fresh");
    assert_eq!(svc.arbiter().instances(), 1);
    assert_eq!(worker.joins().len(), 2);

    // Second recoverable failure: the budget (1) is consumed → escalates typed-terminal.
    svc.handle_worker_event(&terminated(
        "run-r",
        second_gen,
        TerminalOutcome::FailedRetryable {
            reason: "transport loss again".into(),
        },
    ))
    .unwrap();
    let row = svc.store().get_run("run-r").unwrap().unwrap();
    assert_eq!(row.run_state, RunState::FailedTerminal);
    assert!(row
        .terminal_reason
        .as_deref()
        .is_some_and(|r| r.contains("retry budget exhausted")));
    // Terminal: the tick reconverges nothing and no further join is issued.
    assert_eq!(svc.reconcile_tick().await.unwrap(), 0);
    assert_eq!(worker.joins().len(), 2);
}

/// The crash window between observed teardown and the terminal commit is repaired by the startup
/// pass: the marker's target commits and a completed run is not reconverged.
#[tokio::test]
async fn crash_window_repair_finishes_pending_release_on_start() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vhc.db");
    {
        let worker = StreamingWorker::new();
        let svc = service_over(VhcStore::open(&path).unwrap(), worker.clone(), 5);
        svc.vhc_join("run-w".into(), policy(), "op".into())
            .await
            .unwrap();
        // Simulate the crash: teardown observed + marker durable, but the node died before the
        // terminal commit (the instance map / arbiter state died with the process).
        svc.store()
            .begin_release(
                "run-w",
                RunState::Completed,
                Some("module signaled run end"),
            )
            .unwrap();
    }
    let worker = StreamingWorker::new();
    let svc = service_over(VhcStore::open(&path).unwrap(), worker.clone(), 5);
    assert_eq!(
        svc.start().await.unwrap(),
        0,
        "the repaired completed run is not reconverged"
    );
    let row = svc.store().get_run("run-w").unwrap().unwrap();
    assert_eq!(row.run_state, RunState::Completed);
    assert_eq!(row.pending_run_state, None);
    assert!(worker.joins().is_empty());
}

/// A restart with a standing joined intent and a live (`running`) observation reconverges with
/// the RETAINED incarnation — a node restart is not churn (no live predecessor exists).
#[tokio::test]
async fn restart_reconverges_retaining_the_incarnation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vhc.db");
    let first_instance;
    {
        let worker = StreamingWorker::new();
        let svc = service_over(VhcStore::open(&path).unwrap(), worker.clone(), 5);
        svc.vhc_join("run-k".into(), policy(), "op".into())
            .await
            .unwrap();
        first_instance = svc.store().get_run("run-k").unwrap().unwrap().instance;
        assert!(first_instance > 0);
    }
    let worker = StreamingWorker::new();
    let svc = service_over(VhcStore::open(&path).unwrap(), worker.clone(), 5);
    assert_eq!(svc.start().await.unwrap(), 1);
    let row = svc.store().get_run("run-k").unwrap().unwrap();
    assert_eq!(
        row.instance, first_instance,
        "a process restart retains the logical incarnation"
    );
    assert_eq!(row.run_state, RunState::Running);
    assert_eq!(worker.joins(), vec!["run-k"]);
}

/// Transport loss is OBSERVED: a pump stream that closes without its instance's terminal event
/// classifies the instance `failed_retryable` (durable intent preserved) and the tick
/// reconverges it as a fresh incarnation.
#[tokio::test]
async fn stream_closure_classifies_recoverable_and_reconverges() {
    let worker = StreamingWorker::new();
    let svc = service_over(VhcStore::open_in_memory().unwrap(), worker.clone(), 5);
    svc.vhc_join("run-s".into(), policy(), "op".into())
        .await
        .unwrap();
    let first_gen = svc.store().get_run("run-s").unwrap().unwrap().instance;
    assert_eq!(svc.arbiter().instances(), 1);

    // Sever the worker's event stream (crash / transport loss). The pump observes the closure.
    worker.sever_streams();
    wait_for(&svc, "run-s", |s| s == RunState::FailedRetryable).await;
    assert_eq!(svc.arbiter().instances(), 0, "the dead instance released");
    let row = svc.store().get_run("run-s").unwrap().unwrap();
    assert_eq!(row.retry_count, 1);

    // The reconciliation tick reconverges under the durable intent, as a new incarnation.
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    assert_eq!(svc.reconcile_tick().await.unwrap(), 1);
    let row = svc.store().get_run("run-s").unwrap().unwrap();
    assert_eq!(row.run_state, RunState::Running);
    assert!(row.instance > first_gen);
    assert_eq!(worker.joins().len(), 2);
}

/// A leave that ends the stream is NOT transport loss: the instance left the map before the
/// stream closed, so no recoverable failure is synthesized and nothing reconverges.
#[tokio::test]
async fn leave_stream_closure_is_not_classified_as_a_failure() {
    let worker = StreamingWorker::new();
    let svc = service_over(VhcStore::open_in_memory().unwrap(), worker.clone(), 5);
    svc.vhc_join("run-l".into(), policy(), "op".into())
        .await
        .unwrap();
    svc.vhc_leave(
        "run-l".into(),
        daemon_api::VhcLeaveMode::Graceful,
        "op2".into(),
    )
    .await
    .unwrap();
    assert_eq!(svc.arbiter().instances(), 0);
    // The leave cleared the pump sink / the stream ends now — no failure may be synthesized.
    worker.sever_streams();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let row = svc.store().get_run("run-l").unwrap().unwrap();
    assert_eq!(row.run_state, RunState::Left);
    assert_eq!(row.retry_count, 0);
    assert_eq!(svc.reconcile_tick().await.unwrap(), 0, "left never retries");
}
