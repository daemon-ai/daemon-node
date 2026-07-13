// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `SwarmService` + `swarm.db` unit tests (W1): event fanout, durable join-intent persistence +
//! reload re-convergence, `swarm.db` migration idempotence, and disabled-by-default (no worker spawn
//! when `enabled = false`). The worker is a trait-level `FakeWorker` (the `WorkerControl` seam) — no
//! subprocess — recording every call so we can assert the service never touches the worker while
//! disabled and re-issues exactly the persisted intents on restart.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{
    NodeEvent, SwarmApi, SwarmEligibility, SwarmEvent, SwarmLeaveMode, SwarmPolicy, SwarmPolicyMode,
};
use daemon_swarm_node::service::{NodeFeed, SwarmError, WorkerControl};
use daemon_swarm_node::{SwarmService, SwarmServiceParts, SwarmStore, EVENT_WINDOW};
use daemon_swarm_run::config::SwarmConfig;
use daemon_swarm_run::protocol::{self, Eligibility, ErrorClass, Hardware, JoinPolicy, LeaveMode};
use futures::StreamExt;

/// A recording fake of the worker-control seam.
#[derive(Default)]
struct Calls {
    joins: Vec<String>,
    leaves: Vec<String>,
    throttles: usize,
    probes: usize,
    /// The args of the most recent `throttle` (the §10.5 governor lever): `(vram_cap_mb,
    /// duty_cycle_pct, paused)`.
    last_throttle: Option<(Option<u32>, Option<u8>, bool)>,
}

struct FakeWorker {
    calls: Mutex<Calls>,
    hardware: Hardware,
}

impl FakeWorker {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Calls::default()),
            hardware: Hardware {
                gpus: 1,
                vram_mb: 24_000,
                ram_mb: 64_000,
                backend_lanes: vec!["cpu".into()],
                up_kbps: 1_000,
                down_kbps: 1_000,
                disk_free_mb: 100_000,
                throughput_class: "c2".into(),
                ..Default::default()
            },
        })
    }
    fn calls(&self) -> std::sync::MutexGuard<'_, Calls> {
        self.calls.lock().unwrap()
    }
}

#[async_trait]
impl WorkerControl for FakeWorker {
    async fn probe(&self) -> Result<Hardware, SwarmError> {
        self.calls().probes += 1;
        Ok(self.hardware.clone())
    }
    async fn assess(&self, _envelope: Vec<u8>) -> Result<Eligibility, SwarmError> {
        Ok(Eligibility {
            eligible: true,
            reasons: vec![],
            headroom: vec![],
        })
    }
    async fn join(
        &self,
        run_id: String,
        _coordinator: String,
        _credentials: Vec<u8>,
        _policy: JoinPolicy,
    ) -> Result<(), SwarmError> {
        self.calls().joins.push(run_id);
        Ok(())
    }
    async fn leave(&self, run_id: String, _mode: LeaveMode) -> Result<(), SwarmError> {
        self.calls().leaves.push(run_id);
        Ok(())
    }
    async fn throttle(
        &self,
        vram_cap_mb: Option<u32>,
        duty_cycle_pct: Option<u8>,
        paused: bool,
    ) -> Result<(), SwarmError> {
        let mut c = self.calls();
        c.throttles += 1;
        c.last_throttle = Some((vram_cap_mb, duty_cycle_pct, paused));
        Ok(())
    }
}

fn enabled_config() -> SwarmConfig {
    SwarmConfig {
        enabled: true,
        ..SwarmConfig::default()
    }
}

fn policy() -> SwarmPolicy {
    SwarmPolicy {
        mode: SwarmPolicyMode::Idle,
        vram_cap_mb: 8_000,
        duty_cycle_pct: 90,
        schedule: None,
    }
}

fn service(config: SwarmConfig, worker: Arc<FakeWorker>, feed: Option<NodeFeed>) -> SwarmService {
    SwarmService::new(SwarmServiceParts {
        config,
        store: SwarmStore::open_in_memory().unwrap(),
        worker,
        feed,
    })
}

#[tokio::test]
async fn disabled_by_default_never_touches_worker() {
    let worker = FakeWorker::new();
    let svc = service(SwarmConfig::default(), worker.clone(), None);
    assert!(!svc.enabled(), "swarm is off by default (§10.6)");
    // start() must be a no-op: no re-convergence, no probe, no join.
    assert_eq!(svc.start().await.unwrap(), 0);
    // Every worker-touching API op resolves to Unsupported (disabled), spawning nothing.
    assert!(matches!(
        svc.swarm_join("r1".into(), policy(), "op".into()).await,
        Err(daemon_api::ApiError::Unsupported(_))
    ));
    assert!(matches!(
        svc.swarm_hardware_report().await,
        Err(daemon_api::ApiError::Unsupported(_))
    ));
    let c = worker.calls();
    assert_eq!(c.joins.len(), 0);
    assert_eq!(c.probes, 0);
    assert_eq!(c.throttles, 0);
}

#[tokio::test]
async fn join_persists_and_reload_reconverges() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm.db");

    // First boot: enabled service joins two runs, then leaves one.
    {
        let worker = FakeWorker::new();
        let svc = SwarmService::new(SwarmServiceParts {
            config: enabled_config(),
            store: SwarmStore::open(&path).unwrap(),
            worker: worker.clone(),
            feed: None,
        });
        svc.swarm_join("run-a".into(), policy(), "op-a".into())
            .await
            .unwrap();
        svc.swarm_join("run-b".into(), policy(), "op-b".into())
            .await
            .unwrap();
        svc.swarm_leave("run-b".into(), SwarmLeaveMode::Graceful, "op-c".into())
            .await
            .unwrap();
        assert_eq!(worker.calls().joins, vec!["run-a", "run-b"]);
        assert_eq!(worker.calls().leaves, vec!["run-b"]);
    }

    // Second boot: a fresh worker + service over the SAME swarm.db. start() re-issues JoinRun for
    // the one still-active intent (run-a), not the left one (run-b) — durable re-convergence.
    {
        let worker = FakeWorker::new();
        let svc = SwarmService::new(SwarmServiceParts {
            config: enabled_config(),
            store: SwarmStore::open(&path).unwrap(),
            worker: worker.clone(),
            feed: None,
        });
        let rejoined = svc.start().await.unwrap();
        assert_eq!(rejoined, 1, "only the active intent re-converges");
        assert_eq!(worker.calls().joins, vec!["run-a"]);
        // The run list still shows both rows (run-b retained, marked not-joined).
        let mut runs = svc.swarm_run_list().await.unwrap();
        runs.sort_by(|a, b| a.run_id.cmp(&b.run_id));
        assert_eq!(runs.len(), 2);
        assert!(runs[0].joined && runs[0].run_id == "run-a");
        assert!(!runs[1].joined && runs[1].run_id == "run-b");
        // Eligibility is node-computed (from the probe) and mirrored on the row (ADR-003).
        assert!(runs[0].eligibility.eligible);
    }
}

#[tokio::test]
async fn event_fanout_persists_broadcasts_and_pings_feed() {
    let feed_log: Arc<Mutex<Vec<NodeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let feed_log2 = feed_log.clone();
    let feed: NodeFeed = Arc::new(move |ev: NodeEvent| feed_log2.lock().unwrap().push(ev));
    let worker = FakeWorker::new();
    let svc = service(enabled_config(), worker, Some(feed));

    // Events arrive for a joined run — join first (creates the run row), then reset the feed log so
    // we count only the event-driven SwarmChanged pings below.
    svc.swarm_join("run-1".into(), policy(), "op".into())
        .await
        .unwrap();
    feed_log.lock().unwrap().clear();

    // A live subscriber for run-1.
    let mut sub = svc.swarm_subscribe(Some("run-1".into())).await.unwrap();

    // Feed a worker phase → progress → outcome → error sequence.
    let outs = svc
        .handle_worker_event(&protocol::Event::RunPhase {
            run_id: "run-1".into(),
            phase: "RoundTrain".into(),
            epoch: 1,
            round: 5,
        })
        .unwrap();
    assert!(matches!(
        outs.as_slice(),
        [SwarmEvent::Phase { round: 5, .. }]
    ));

    let outs = svc
        .handle_worker_event(&protocol::Event::RoundProgress {
            inner_step: 2,
            loss: 3.5,
            tokens_per_s: 12.0,
            up_bytes: 100,
            down_bytes: 200,
            peers: 3,
        })
        .unwrap();
    assert!(matches!(
        outs.as_slice(),
        [SwarmEvent::Progress {
            loss_micros: 3_500_000,
            peers: 3,
            ..
        }]
    ));

    svc.handle_worker_event(&protocol::Event::RoundOutcome {
        round: 5,
        committed: 3,
        ingested: 3,
        stalled: false,
        digest: [0u8; 16],
    })
    .unwrap();

    svc.handle_worker_event(&protocol::Event::Error {
        class: ErrorClass::Desync,
        detail: "mismatch".into(),
    })
    .unwrap();

    // The subscriber sees the four run-1 events in order.
    let kinds: Vec<&str> = collect(&mut sub, 4)
        .await
        .iter()
        .map(SwarmEvent::kind)
        .collect();
    assert_eq!(kinds, ["phase", "progress", "round_outcome", "error"]);

    // Each handled worker event pinged the node feed with a SwarmChanged pointer. Scope the guard so
    // it never crosses the await below.
    {
        let feed_events = feed_log.lock().unwrap();
        assert_eq!(feed_events.len(), 4);
        assert!(feed_events.iter().all(|e| matches!(
            e,
            NodeEvent::SwarmChanged { run_id: Some(r), .. } if r == "run-1"
        )));
    }

    // Contribution folded from the events (one non-stalled round, bytes from progress).
    let detail = svc.swarm_run_detail("run-1".into()).await.unwrap().unwrap();
    assert_eq!(detail.contribution.rounds, 1);
    assert_eq!(detail.contribution.bytes_up, 100);
    assert_eq!(detail.contribution.bytes_down, 200);
    // All four events are in the windowed log (newest last).
    assert_eq!(detail.recent_events.len(), 4);
}

#[tokio::test]
async fn governor_throttle_lever_reaches_worker_with_combined_budget_clamp() {
    // §10.5 governor drill (B3): a synthetic inference-pressure signal arrives as a policy update
    // clamping the swarm's budget (on a unified box `vram_cap_mb` clamps the *combined* device+host
    // budget — Merge-2 spec-amendment #1). `swarm_set_policy` must push that lever through to the
    // worker's `throttle` verbatim, so the co-resident inference tenant is protected.
    let worker = FakeWorker::new();
    let svc = service(enabled_config(), worker.clone(), None);

    let pressure = SwarmPolicy {
        mode: SwarmPolicyMode::Idle,
        vram_cap_mb: 4_096, // clamp the combined budget under inference pressure
        duty_cycle_pct: 25, // and throttle the duty cycle
        schedule: None,
    };
    svc.swarm_set_policy(pressure).await.unwrap();

    let c = worker.calls();
    assert_eq!(
        c.throttles, 1,
        "the governor lever reached the worker exactly once"
    );
    assert_eq!(
        c.last_throttle,
        Some((Some(4_096), Some(25), false)),
        "the vram cap (combined-budget clamp) + duty cycle are forwarded verbatim (§10.5)"
    );
}

#[tokio::test]
async fn checkpoint_published_yields_contribution_event_and_credit() {
    let worker = FakeWorker::new();
    let svc = service(enabled_config(), worker, None);
    // Establish the current run first.
    svc.handle_worker_event(&protocol::Event::RunPhase {
        run_id: "run-x".into(),
        phase: "witness".into(),
        epoch: 0,
        round: 1,
    })
    .unwrap();
    let outs = svc
        .handle_worker_event(&protocol::Event::CheckpointPublished {
            round: 1,
            hash: "abc".into(),
            location: "r2://x".into(),
        })
        .unwrap();
    // CheckpointPublished emits a Contribution event carrying the fresh totals (1 credit).
    assert!(matches!(
        outs.as_slice(),
        [SwarmEvent::Contribution { contribution, .. }] if contribution.checkpoint_credits == 1
    ));
}

#[test]
fn swarm_db_migration_is_idempotent_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swarm.db");
    let elig = SwarmEligibility::default();
    {
        let store = SwarmStore::open(&path).unwrap();
        store
            .put_join_intent("r1", "coord", &policy(), None, &elig)
            .unwrap();
    }
    // Re-opening re-runs the migration ladder (a no-op at the same user_version) and the row is
    // still there — proving migrations are idempotent + durable.
    let store = SwarmStore::open(&path).unwrap();
    assert_eq!(store.get_run("r1").unwrap().unwrap().run_id, "r1");
    // A third open is still fine (idempotence again).
    drop(store);
    let store = SwarmStore::open(&path).unwrap();
    assert_eq!(store.list_runs().unwrap().len(), 1);
}

#[test]
fn swarm_events_log_is_windowed() {
    let store = SwarmStore::open_in_memory().unwrap();
    for i in 0..(EVENT_WINDOW + 50) {
        store
            .append_event(&SwarmEvent::Phase {
                run_id: "r1".into(),
                phase: format!("p{i}"),
                epoch: 0,
                round: i as u64,
            })
            .unwrap();
    }
    // The ring is capped at EVENT_WINDOW; the newest entries are retained.
    assert_eq!(store.event_count("r1").unwrap(), EVENT_WINDOW);
    let recent = store.recent_events("r1", EVENT_WINDOW).unwrap();
    assert_eq!(recent.len(), EVENT_WINDOW);
    // Chronological order (oldest → newest); the last is the highest round.
    if let SwarmEvent::Phase { round, .. } = recent.last().unwrap() {
        assert_eq!(*round, (EVENT_WINDOW + 49) as u64);
    } else {
        panic!("expected Phase");
    }
}

/// Drain `n` items from a subscription stream (with a timeout so a bug can't hang the test).
async fn collect(sub: &mut daemon_api::SwarmEventStream, n: usize) -> Vec<SwarmEvent> {
    let mut out = Vec::new();
    for _ in 0..n {
        match tokio::time::timeout(std::time::Duration::from_secs(2), sub.next()).await {
            Ok(Some(ev)) => out.push(ev),
            _ => break,
        }
    }
    out
}
