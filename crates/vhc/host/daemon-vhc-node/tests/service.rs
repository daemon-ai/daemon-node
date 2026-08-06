// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `VhcService` + `vhc.db` unit tests: event fanout, durable join-intent persistence +
//! reload re-convergence, `vhc.db` migration idempotence, and disabled-by-default (no worker spawn
//! when `enabled = false`). The worker is a trait-level `FakeWorker` (the `WorkerControl` seam) — no
//! subprocess — recording every call so we can assert the service never touches the worker while
//! disabled and re-issues exactly the persisted intents on restart.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{
    NodeEvent, VhcApi, VhcEligibility, VhcEvent, VhcLeaveMode, VhcPolicy, VhcPolicyMode,
};
use daemon_vhc_node::service::{NodeFeed, VhcError, WorkerControl};
use daemon_vhc_node::{
    DiscoveredRun, RunDiscovery, VhcService, VhcServiceParts, VhcStore, EVENT_WINDOW,
};
use daemon_vhc_session::config::VhcConfig;
use daemon_vhc_session::protocol::{
    self, Eligibility, ErrorClass, Hardware, JoinPolicy, LeaveMode,
};
use futures::StreamExt;

/// A recording fake of the worker-control seam.
#[derive(Default)]
struct Calls {
    joins: Vec<String>,
    leaves: Vec<String>,
    throttles: usize,
    probes: usize,
    /// The envelope bytes passed to each `assess` (proves the join flow fetched + assessed).
    assessed_envelopes: Vec<Vec<u8>>,
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
    async fn probe(&self) -> Result<Hardware, VhcError> {
        self.calls().probes += 1;
        Ok(self.hardware.clone())
    }
    async fn assess(
        &self,
        envelope: Vec<u8>,
        _role: Option<String>,
    ) -> Result<Eligibility, VhcError> {
        self.calls().assessed_envelopes.push(envelope);
        // A distinctive verdict so a test can tell the §6.5 assess path from the probe fallback — plus
        // the composed reservation the ledger charges. A verdict with no resource figure is a typed
        // `EstimateNotComposable` refusal now that the owner-cap fallback is gone (`d9a32ab8`; arbiter-charge
        // disposition, 2026-07-26), so a drill about the discovery path has to state a need in order to
        // reach the discovery path at all.
        Ok(Eligibility {
            eligible: true,
            reasons: vec!["assessed against envelope".into()],
            headroom: vec![
                ("assessed_micro_batch".into(), 64),
                (
                    daemon_vhc_abi::RESERVATION_DEVICE_BYTES_KEY.into(),
                    256 << 20,
                ),
                (daemon_vhc_abi::RESERVATION_HOST_BYTES_KEY.into(), 512 << 20),
            ],
            refusal_code: None,
            admitted_tuple: None,
        })
    }
    async fn join(
        &self,
        run_id: String,
        _coordinator: String,
        _credentials: Vec<u8>,
        _policy: JoinPolicy,
        _admitted_tuple: Option<daemon_vhc_session::protocol::AdmittedTuple>,
    ) -> Result<(), VhcError> {
        self.calls().joins.push(run_id);
        Ok(())
    }
    async fn leave(&self, run_id: String, _mode: LeaveMode) -> Result<(), VhcError> {
        self.calls().leaves.push(run_id);
        Ok(())
    }
    async fn throttle(
        &self,
        vram_cap_mb: Option<u32>,
        duty_cycle_pct: Option<u8>,
        paused: bool,
    ) -> Result<(), VhcError> {
        let mut c = self.calls();
        c.throttles += 1;
        c.last_throttle = Some((vram_cap_mb, duty_cycle_pct, paused));
        Ok(())
    }
}

fn enabled_config() -> VhcConfig {
    VhcConfig {
        enabled: true,
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

fn service(config: VhcConfig, worker: Arc<FakeWorker>, feed: Option<NodeFeed>) -> VhcService {
    VhcService::new(VhcServiceParts {
        config,
        store: VhcStore::open_in_memory().unwrap(),
        worker,
        feed,
        discovery: None,
        budget: None,
        worker_factory: None,
        identity_dir: None,
        run_dir: None,
        seat_directory: None,
    })
}

#[tokio::test]
async fn disabled_by_default_never_touches_worker() {
    let worker = FakeWorker::new();
    let svc = service(VhcConfig::default(), worker.clone(), None);
    assert!(!svc.enabled(), "vhc is off by default (§10.6)");
    // start() must be a no-op: no re-convergence, no probe, no join.
    assert_eq!(svc.start().await.unwrap(), 0);
    // Every worker-touching API op resolves to Unsupported (disabled), spawning nothing.
    assert!(matches!(
        svc.vhc_join("r1".into(), policy(), "op".into()).await,
        Err(daemon_api::ApiError::Unsupported(_))
    ));
    assert!(matches!(
        svc.vhc_hardware_report().await,
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
    let path = dir.path().join("vhc.db");

    // First boot: enabled service joins two runs, then leaves one.
    {
        let worker = FakeWorker::new();
        let svc = VhcService::new(VhcServiceParts {
            config: enabled_config(),
            store: VhcStore::open(&path).unwrap(),
            worker: worker.clone(),
            feed: None,
            discovery: None,
            budget: None,
            worker_factory: None,
            identity_dir: None,
            run_dir: None,
            seat_directory: None,
        });
        svc.vhc_join("run-a".into(), policy(), "op-a".into())
            .await
            .unwrap();
        svc.vhc_join("run-b".into(), policy(), "op-b".into())
            .await
            .unwrap();
        svc.vhc_leave("run-b".into(), VhcLeaveMode::Graceful, "op-c".into())
            .await
            .unwrap();
        assert_eq!(worker.calls().joins, vec!["run-a", "run-b"]);
        assert_eq!(worker.calls().leaves, vec!["run-b"]);
    }

    // Second boot: a fresh worker + service over the SAME vhc.db. start() re-issues JoinRun for
    // the one still-active intent (run-a), not the left one (run-b) — durable re-convergence.
    {
        let worker = FakeWorker::new();
        let svc = VhcService::new(VhcServiceParts {
            config: enabled_config(),
            store: VhcStore::open(&path).unwrap(),
            worker: worker.clone(),
            feed: None,
            discovery: None,
            budget: None,
            worker_factory: None,
            identity_dir: None,
            run_dir: None,
            seat_directory: None,
        });
        let rejoined = svc.start().await.unwrap();
        assert_eq!(rejoined, 1, "only the active intent re-converges");
        assert_eq!(worker.calls().joins, vec!["run-a"]);
        // The run list still shows both rows (run-b retained, marked not-joined).
        let mut runs = svc.vhc_run_list().await.unwrap();
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
    // we count only the event-driven VhcChanged pings below.
    svc.vhc_join("run-1".into(), policy(), "op".into())
        .await
        .unwrap();
    feed_log.lock().unwrap().clear();

    // A live subscriber for run-1.
    let mut sub = svc.vhc_subscribe(Some("run-1".into())).await.unwrap();

    // Feed a worker phase → progress → outcome → error sequence.
    let outs = svc
        .handle_worker_event(&protocol::Event::RunPhase {
            run_id: "run-1".into(),
            phase: "RoundTrain".into(),
            epoch: 1,
            round: 5,
            generation: 1,
        })
        .unwrap();
    assert!(matches!(
        outs.as_slice(),
        [VhcEvent::Phase { round: 5, .. }]
    ));

    let outs = svc
        .handle_worker_event(&protocol::Event::RoundProgress {
            inner_step: 2,
            loss: 3.5,
            tokens_per_s: 12.0,
            up_bytes: 100,
            down_bytes: 200,
            peers: 3,
            generation: 1,
        })
        .unwrap();
    assert!(matches!(
        outs.as_slice(),
        [VhcEvent::Progress {
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
        generation: 1,
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
        .map(VhcEvent::kind)
        .collect();
    assert_eq!(kinds, ["phase", "progress", "round_outcome", "error"]);

    // Each handled worker event pinged the node feed with a VhcChanged pointer. Scope the guard so
    // it never crosses the await below.
    {
        let feed_events = feed_log.lock().unwrap();
        assert_eq!(feed_events.len(), 4);
        assert!(feed_events.iter().all(|e| matches!(
            e,
            NodeEvent::VhcChanged { run_id: Some(r), .. } if r == "run-1"
        )));
    }

    // Contribution folded from the events (one non-stalled round, bytes from progress).
    let detail = svc.vhc_run_detail("run-1".into()).await.unwrap().unwrap();
    assert_eq!(detail.contribution.rounds, 1);
    assert_eq!(detail.contribution.bytes_up, 100);
    assert_eq!(detail.contribution.bytes_down, 200);
    // All four events are in the windowed log (newest last).
    assert_eq!(detail.recent_events.len(), 4);
}

#[tokio::test]
async fn governor_throttle_lever_reaches_worker_with_combined_budget_clamp() {
    // §10.5 governor drill (B3): a synthetic inference-pressure signal arrives as a policy update
    // clamping the vhc's budget (on a unified box `vram_cap_mb` clamps the *combined* device+host
    // budget — Merge-2 spec-amendment #1). `vhc_set_policy` must push that lever through to the
    // worker's `throttle` verbatim, so the co-resident inference tenant is protected.
    let worker = FakeWorker::new();
    let svc = service(enabled_config(), worker.clone(), None);

    let pressure = VhcPolicy {
        mode: VhcPolicyMode::Idle,
        vram_cap_mb: 4_096, // clamp the combined budget under inference pressure
        duty_cycle_pct: 25, // and throttle the duty cycle
        schedule: None,
    };
    svc.vhc_set_policy(pressure).await.unwrap();

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
        generation: 1,
    })
    .unwrap();
    let outs = svc
        .handle_worker_event(&protocol::Event::CheckpointPublished {
            round: 1,
            hash: "abc".into(),
            location: "r2://x".into(),
            generation: 1,
            kind: "drain".into(),
        })
        .unwrap();
    // CheckpointPublished emits a Contribution event carrying the fresh totals (1 credit).
    assert!(matches!(
        outs.as_slice(),
        [VhcEvent::Contribution { contribution, .. }] if contribution.checkpoint_credits == 1
    ));
}

#[test]
fn vhc_db_migration_is_idempotent_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vhc.db");
    let elig = VhcEligibility::default();
    {
        let store = VhcStore::open(&path).unwrap();
        store
            .put_join_intent("r1", "coord", &policy(), None, &elig)
            .unwrap();
    }
    // Re-opening re-runs the migration ladder (a no-op at the same user_version) and the row is
    // still there — proving migrations are idempotent + durable.
    let store = VhcStore::open(&path).unwrap();
    assert_eq!(store.get_run("r1").unwrap().unwrap().run_id, "r1");
    // A third open is still fine (idempotence again).
    drop(store);
    let store = VhcStore::open(&path).unwrap();
    assert_eq!(store.list_runs().unwrap().len(), 1);
}

#[test]
fn vhc_events_log_is_windowed() {
    let store = VhcStore::open_in_memory().unwrap();
    for i in 0..(EVENT_WINDOW + 50) {
        store
            .append_event(&VhcEvent::Phase {
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
    if let VhcEvent::Phase { round, .. } = recent.last().unwrap() {
        assert_eq!(*round, (EVENT_WINDOW + 49) as u64);
    } else {
        panic!("expected Phase");
    }
}

/// A fake discovery seam: resolves one run to a fixed coordinator + envelope, recording the calls.
struct FakeDiscovery {
    coordinator: String,
    envelope: Vec<u8>,
}

#[async_trait]
impl RunDiscovery for FakeDiscovery {
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, VhcError> {
        Ok(vec![self.run("run-disc")])
    }
    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, VhcError> {
        Ok(Some(self.run(run_id)))
    }
    async fn fetch_envelope(&self, _run_id: &str) -> Result<Vec<u8>, VhcError> {
        Ok(self.envelope.clone())
    }
}

impl FakeDiscovery {
    fn run(&self, run_id: &str) -> DiscoveredRun {
        DiscoveredRun {
            run_id: run_id.to_string(),
            coordinator: self.coordinator.clone(),
            envelope_hash: "deadbeef".into(),
            proto_version: 3,
        }
    }
}

/// With a discovery seam, `vhc_join` discovers the run, fetches the frozen envelope, and derives
/// eligibility from the worker's real §6.5 `AssessRun` (not the probe), taking the coordinator from
/// discovery — the A1 join flow.
#[tokio::test]
async fn join_discovers_fetches_envelope_and_assesses() {
    let worker = FakeWorker::new();
    let discovery = Arc::new(FakeDiscovery {
        coordinator: "https://coord.example/api/v1/vhc".into(),
        envelope: b"frozen-envelope-bytes".to_vec(),
    });
    // The owner allowlists the discovered coordinator's base (spec §11.1) — the join proceeds.
    let config = daemon_vhc_session::config::VhcConfig {
        coordinator_allowlist: vec!["https://coord.example/api/v1/vhc".into()],
        ..enabled_config()
    };
    let svc = VhcService::new(VhcServiceParts {
        config,
        store: VhcStore::open_in_memory().unwrap(),
        worker: worker.clone(),
        feed: None,
        discovery: Some(discovery),
        budget: None,
        worker_factory: None,
        identity_dir: None,
        run_dir: None,
        seat_directory: None,
    });

    svc.vhc_join("run-disc".into(), policy(), "op".into())
        .await
        .unwrap();

    // The envelope was fetched and handed to AssessRun; the probe path was NOT taken.
    {
        let c = worker.calls();
        assert_eq!(
            c.assessed_envelopes,
            vec![b"frozen-envelope-bytes".to_vec()]
        );
        assert_eq!(
            c.probes, 0,
            "assess supersedes the probe when discovery is configured"
        );
        assert_eq!(c.joins, vec!["run-disc"]);
    }

    // The persisted run carries the discovery coordinator + the assess-derived eligibility.
    let detail = svc
        .vhc_run_detail("run-disc".into())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(detail.coordinator, "https://coord.example/api/v1/vhc");
    assert!(detail.summary.eligibility.eligible);
    assert_eq!(
        detail
            .summary
            .eligibility
            .headroom
            .get("assessed_micro_batch"),
        Some(&64),
        "eligibility came from AssessRun, not the hardware probe"
    );
}

/// A discovered coordinator OUTSIDE the owner's allowlist is a typed refusal BEFORE anything
/// reaches the worker (spec §11.1): no envelope fetch, no assess, no join, nothing persisted.
#[tokio::test]
async fn join_refuses_a_coordinator_outside_the_allowlist() {
    let worker = FakeWorker::new();
    let discovery = Arc::new(FakeDiscovery {
        coordinator: "https://rogue.example/api/v1/vhc".into(),
        envelope: b"frozen-envelope-bytes".to_vec(),
    });
    // The default allowlist names only the product coordinator — rogue.example is not on it.
    let svc = VhcService::new(VhcServiceParts {
        config: enabled_config(),
        store: VhcStore::open_in_memory().unwrap(),
        worker: worker.clone(),
        feed: None,
        discovery: Some(discovery),
        budget: None,
        worker_factory: None,
        identity_dir: None,
        run_dir: None,
        seat_directory: None,
    });

    let err = svc
        .vhc_join("run-disc".into(), policy(), "op".into())
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("allowlist"),
        "typed allowlist refusal, got: {err}"
    );
    // Fail closed: the worker was never consulted and no intent was persisted.
    {
        let c = worker.calls();
        assert!(
            c.assessed_envelopes.is_empty(),
            "no envelope reached assess"
        );
        assert!(c.joins.is_empty(), "no join was issued");
    }
    assert!(svc
        .vhc_run_detail("run-disc".into())
        .await
        .unwrap()
        .is_none());
}

/// Drain `n` items from a subscription stream (with a timeout so a bug can't hang the test).
async fn collect(sub: &mut daemon_api::VhcEventStream, n: usize) -> Vec<VhcEvent> {
    let mut out = Vec::new();
    for _ in 0..n {
        match tokio::time::timeout(std::time::Duration::from_secs(2), sub.next()).await {
            Ok(Some(ev)) => out.push(ev),
            _ => break,
        }
    }
    out
}

/// The late-join checkpoint seam (spec §9; lane R): a run discovery that records published
/// pointers per `(role, kind)` slot and resolves the role-scoped best — the round-trip the
/// node's publish hook + join-time restore resolution drive. The default trait methods are
/// overridden here exactly as the production `EgressRunDiscovery` overrides them over the
/// registry.
#[derive(Default)]
struct CheckpointDiscovery {
    published: Mutex<Vec<daemon_vhc_node::CheckpointPointer>>,
}

#[async_trait]
impl RunDiscovery for CheckpointDiscovery {
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, VhcError> {
        Ok(Vec::new())
    }
    async fn get_run(&self, _run_id: &str) -> Result<Option<DiscoveredRun>, VhcError> {
        Ok(None)
    }
    async fn fetch_envelope(&self, _run_id: &str) -> Result<Vec<u8>, VhcError> {
        Ok(Vec::new())
    }
    async fn publish_checkpoint(
        &self,
        _run_id: &str,
        role: &str,
        kind: &str,
        round: u64,
        hash: &str,
        size: u64,
    ) -> Result<(), VhcError> {
        self.published
            .lock()
            .unwrap()
            .push(daemon_vhc_node::CheckpointPointer {
                role: role.to_string(),
                kind: kind.to_string(),
                round,
                hash: hash.to_string(),
                size,
            });
        Ok(())
    }
    async fn fetch_checkpoint(
        &self,
        _run_id: &str,
        role: &str,
    ) -> Result<Option<daemon_vhc_node::CheckpointPointer>, VhcError> {
        Ok(daemon_vhc_node::best_restore_pointer(
            &self.published.lock().unwrap(),
            role,
        ))
    }
}

/// A discovery seam resolving one run (allowlisted coordinator + envelope) AND serving
/// checkpoint pointers — the join-time restore-freshness judgment's input surface.
struct StaleCheckpointDiscovery {
    trainer_round: u64,
    coordinator_round: u64,
}

#[async_trait]
impl RunDiscovery for StaleCheckpointDiscovery {
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, VhcError> {
        Ok(Vec::new())
    }
    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, VhcError> {
        Ok(Some(DiscoveredRun {
            run_id: run_id.to_string(),
            coordinator: "https://coord.example/api/v1/vhc".into(),
            envelope_hash: "deadbeef".into(),
            proto_version: 3,
        }))
    }
    async fn fetch_envelope(&self, _run_id: &str) -> Result<Vec<u8>, VhcError> {
        Ok(b"frozen-envelope-bytes".to_vec())
    }
    async fn fetch_checkpoint(
        &self,
        _run_id: &str,
        role: &str,
    ) -> Result<Option<daemon_vhc_node::CheckpointPointer>, VhcError> {
        let round = match role {
            "trainer" => self.trainer_round,
            "coordinator" => self.coordinator_round,
            _ => return Ok(None),
        };
        Ok(Some(daemon_vhc_node::CheckpointPointer {
            role: role.to_string(),
            kind: "live".into(),
            round,
            hash: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262".into(),
            size: 1024,
        }))
    }
}

/// **Join-time cadence-vs-ring reachability** (the recovery-honesty check): a trainer whose
/// freshest restore pointer is more than the retained record horizon behind the run's live
/// head is refused TYPED at join — before any rehydration — instead of wedging into the
/// module's post-restore `GapRefused`. A fence within the horizon joins normally.
#[tokio::test]
async fn join_refuses_a_checkpoint_too_stale_for_the_retained_horizon() {
    let config = VhcConfig {
        coordinator_allowlist: vec!["https://coord.example/api/v1/vhc".into()],
        ..enabled_config()
    };

    // Restored fence 1 vs live head 10: 9 rounds behind, horizon 4 — refused typed.
    let worker = FakeWorker::new();
    let svc = VhcService::new(VhcServiceParts {
        config: config.clone(),
        store: VhcStore::open_in_memory().unwrap(),
        worker: worker.clone(),
        feed: None,
        discovery: Some(Arc::new(StaleCheckpointDiscovery {
            trainer_round: 1,
            coordinator_round: 10,
        })),
        budget: None,
        worker_factory: None,
        identity_dir: None,
        run_dir: None,
        seat_directory: None,
    });
    let err = svc
        .vhc_join("run-stale".into(), policy(), "op".into())
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("checkpoint too stale"),
        "typed staleness refusal, got: {err}"
    );
    assert!(worker.calls().joins.is_empty(), "no join was issued");

    // Restored fence 8 vs live head 10: within the horizon — the join proceeds.
    let worker = FakeWorker::new();
    let svc = VhcService::new(VhcServiceParts {
        config,
        store: VhcStore::open_in_memory().unwrap(),
        worker: worker.clone(),
        feed: None,
        discovery: Some(Arc::new(StaleCheckpointDiscovery {
            trainer_round: 8,
            coordinator_round: 10,
        })),
        budget: None,
        worker_factory: None,
        identity_dir: None,
        run_dir: None,
        seat_directory: None,
    });
    svc.vhc_join("run-fresh".into(), policy(), "op".into())
        .await
        .expect("a fence within the horizon joins");
    assert_eq!(worker.calls().joins, vec!["run-fresh".to_string()]);
}

/// The §8.8 recovery surface: one allowlisted run, a REAL frozen genesis (the trust root the
/// heads verify against), a trainer restore pointer, NO coordinator pointer — the only head
/// evidence is the signed archive lineage's committed-round claim.
struct ArchiveHeadDiscovery {
    envelope_wire: Vec<u8>,
    heads: Vec<daemon_vhc_proto::ArchiveHeadRecord>,
    trainer_round: u64,
}

#[async_trait]
impl RunDiscovery for ArchiveHeadDiscovery {
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, VhcError> {
        Ok(Vec::new())
    }
    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, VhcError> {
        Ok(Some(DiscoveredRun {
            run_id: run_id.to_string(),
            coordinator: "https://coord.example/api/v1/vhc".into(),
            envelope_hash: "deadbeef".into(),
            proto_version: 3,
        }))
    }
    async fn fetch_envelope(&self, _run_id: &str) -> Result<Vec<u8>, VhcError> {
        Ok(self.envelope_wire.clone())
    }
    async fn fetch_checkpoint(
        &self,
        _run_id: &str,
        role: &str,
    ) -> Result<Option<daemon_vhc_node::CheckpointPointer>, VhcError> {
        if role != "trainer" {
            return Ok(None); // no coordinator pointer — the archive claim is the only evidence
        }
        Ok(Some(daemon_vhc_node::CheckpointPointer {
            role: role.to_string(),
            kind: "live".into(),
            round: self.trainer_round,
            hash: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262".into(),
            size: 1024,
        }))
    }
    async fn fetch_archive_heads(
        &self,
        _run_id: &str,
    ) -> Result<Vec<daemon_vhc_proto::ArchiveHeadRecord>, VhcError> {
        Ok(self.heads.clone())
    }
}

/// A minimal VALID frozen genesis (coordinator + trainer roles, fixture execution requirements)
/// whose coordinator identity is `base`'s peer id — the trust root [`ArchiveHeadDiscovery`]'s
/// heads chain to. Returns the `SignedEnvelope` wire bytes and the run's cryptographic id.
fn frozen_genesis_wire(base: &daemon_vhc_proto::SigningKey) -> (Vec<u8>, daemon_vhc_proto::Hash) {
    use daemon_vhc_proto::{GenesisEnvelope, Identities, RoleEntry, RunSection, SnapshotArtifact};
    use std::collections::BTreeMap;

    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "coord.wasm".to_string(),
        SnapshotArtifact {
            url: "r2://mods/coord.wasm".into(),
            blake3: daemon_vhc_proto::Hash([0xC0; 32]),
            size: None,
        },
    );
    artifacts.insert(
        "trainer.wasm".to_string(),
        SnapshotArtifact {
            url: "r2://mods/trainer.wasm".into(),
            blake3: daemon_vhc_proto::Hash([0x14; 32]),
            size: None,
        },
    );
    let role = |module: &str, lane: &str| RoleEntry {
        lane: lane.into(),
        module: module.into(),
        abi: "vhc@2".into(),
        config: ciborium::value::Value::Map(vec![]),
        grants: daemon_vhc_proto::RoleGrants::default(),
        device_min: daemon_vhc_proto::DeviceMinimums::default(),
        execution: Some(
            daemon_vhc_proto::RoleExecutionRequirements::fixture_over_trivial_plan(vec![
                "cpu".to_string()
            ]),
        ),
    };
    let mut roles = BTreeMap::new();
    roles.insert("coordinator".to_string(), role("coord.wasm", "coordinator"));
    roles.insert("trainer".to_string(), role("trainer.wasm", "trainer"));
    let env = GenesisEnvelope {
        run: RunSection {
            schema: daemon_vhc_proto::GENESIS_SCHEMA_MAJOR,
            run_label: "recovery-drill".into(),
            min_peers: 1,
            max_peers: 8,
            access: daemon_vhc_proto::envelope::Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: None,
        state_contract: None,
        authority: ciborium::value::Value::Map(vec![]),
        transport: daemon_vhc_proto::TransportSelection::default(),
        identities: Identities {
            coordinator: Some(daemon_vhc_proto::peer_id(base)),
            coordinator_set: Vec::new(),
            upgrade_authority: Vec::new(),
        },
    };
    let frozen = env.freeze(base).expect("genesis freezes");
    let run_id = *frozen.run_id();
    let wire = daemon_vhc_proto::SignedEnvelope {
        bytes: frozen.bytes().to_vec(),
        signature: *frozen.signature(),
        signer: *frozen.signer(),
    };
    (
        daemon_vhc_proto::to_canonical_vec(&wire).expect("wire encodes"),
        run_id,
    )
}

/// One base-certified coordinator archive head at chain height 0 carrying `round` as its
/// committed-round freshness claim.
fn coordinator_head(
    base: &daemon_vhc_proto::SigningKey,
    run_id: daemon_vhc_proto::Hash,
    segment_hash: daemon_vhc_proto::Hash,
    round: Option<u64>,
) -> daemon_vhc_proto::ArchiveHeadRecord {
    let run_key = daemon_vhc_proto::SigningKey::from_bytes(&[0x77; 32]);
    let module = daemon_vhc_proto::Hash([0xC0; 32]);
    let cert = daemon_vhc_proto::cert::RunKeyCertificate::issue(
        base,
        daemon_vhc_proto::cert::CertScope {
            run_id,
            epoch: 0,
            role: "coordinator".into(),
            instance: 1,
            module_hash: module,
        },
        daemon_vhc_proto::peer_id(&run_key),
    )
    .expect("cert issues");
    daemon_vhc_proto::ArchiveHeadRecord::publish(
        &run_key,
        cert,
        daemon_vhc_proto::ArchiveHeadBody {
            domain: daemon_vhc_proto::domains::ARCHIVE_HEAD_DOMAIN.to_string(),
            run_id,
            role: "coordinator".into(),
            chain_instance: 1,
            segment: 0,
            segment_hash,
            prev_hash: daemon_vhc_proto::Hash([0; 32]),
            records: 8,
            instance: 1,
            epoch: 0,
            module,
            predecessor: None,
            round,
        },
    )
    .expect("head publishes")
}

/// **The stale-trainer refusal against the TRUE run head** (§8.8; plan phase 5): the registry's
/// coordinator pointer is ABSENT, and the only head evidence is the committed-round claim on the
/// seat lineage's SIGNED archive heads — certificate-chained to the genesis coordinator identity.
/// A trainer whose restore fence is more than the retained horizon behind that verified claim is
/// refused typed at join; a claim at the horizon boundary joins.
#[tokio::test]
async fn join_refuses_a_trainer_stale_against_the_verified_archive_head() {
    let base = daemon_vhc_proto::SigningKey::from_bytes(&[0xB5; 32]);
    let (wire, run_id) = frozen_genesis_wire(&base);
    let config = VhcConfig {
        coordinator_allowlist: vec!["https://coord.example/api/v1/vhc".into()],
        ..enabled_config()
    };
    let horizon = daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS;

    // Fence 1 vs a verified claim of 1 + horizon + 1: beyond reach — refused typed.
    let worker = FakeWorker::new();
    let svc = VhcService::new(VhcServiceParts {
        config: config.clone(),
        store: VhcStore::open_in_memory().unwrap(),
        worker: worker.clone(),
        feed: None,
        discovery: Some(Arc::new(ArchiveHeadDiscovery {
            envelope_wire: wire.clone(),
            heads: vec![coordinator_head(
                &base,
                run_id,
                daemon_vhc_proto::Hash([0xAA; 32]),
                Some(1 + horizon + 1),
            )],
            trainer_round: 1,
        })),
        budget: None,
        worker_factory: None,
        identity_dir: None,
        run_dir: None,
        seat_directory: None,
    });
    let err = svc
        .vhc_join("run-archive-stale".into(), policy(), "op".into())
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("checkpoint too stale"),
        "typed staleness refusal against the verified archive claim, got: {err}"
    );
    assert!(worker.calls().joins.is_empty(), "no join was issued");

    // Fence 1 vs a claim exactly at the horizon boundary: reachable — the join proceeds.
    let worker = FakeWorker::new();
    let svc = VhcService::new(VhcServiceParts {
        config,
        store: VhcStore::open_in_memory().unwrap(),
        worker: worker.clone(),
        feed: None,
        discovery: Some(Arc::new(ArchiveHeadDiscovery {
            envelope_wire: wire,
            heads: vec![coordinator_head(
                &base,
                run_id,
                daemon_vhc_proto::Hash([0xAA; 32]),
                Some(1 + horizon),
            )],
            trainer_round: 1,
        })),
        budget: None,
        worker_factory: None,
        identity_dir: None,
        run_dir: None,
        seat_directory: None,
    });
    svc.vhc_join("run-archive-fresh".into(), policy(), "op".into())
        .await
        .expect("a fence within the verified horizon joins");
    assert_eq!(worker.calls().joins, vec!["run-archive-fresh".to_string()]);
}

/// **The node-side conflicting-head refusal** (§8.8 [AR-4]): two base-certified heads at the
/// same chain height with different content addresses are fork evidence — the join fails CLOSED
/// (typed discovery error), never a silent fresh boot past history that exists.
#[tokio::test]
async fn join_refuses_conflicting_archive_heads_typed() {
    let base = daemon_vhc_proto::SigningKey::from_bytes(&[0xB5; 32]);
    let (wire, run_id) = frozen_genesis_wire(&base);
    let worker = FakeWorker::new();
    let svc = VhcService::new(VhcServiceParts {
        config: VhcConfig {
            coordinator_allowlist: vec!["https://coord.example/api/v1/vhc".into()],
            ..enabled_config()
        },
        store: VhcStore::open_in_memory().unwrap(),
        worker: worker.clone(),
        feed: None,
        discovery: Some(Arc::new(ArchiveHeadDiscovery {
            envelope_wire: wire,
            heads: vec![
                coordinator_head(&base, run_id, daemon_vhc_proto::Hash([0xAA; 32]), Some(2)),
                coordinator_head(&base, run_id, daemon_vhc_proto::Hash([0xBB; 32]), Some(2)),
            ],
            trainer_round: 1,
        })),
        budget: None,
        worker_factory: None,
        identity_dir: None,
        run_dir: None,
        seat_directory: None,
    });
    let err = svc
        .vhc_join("run-forked".into(), policy(), "op".into())
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("do not verify"),
        "typed fork refusal, got: {err}"
    );
    assert!(worker.calls().joins.is_empty(), "no join was issued");
}

#[tokio::test]
async fn checkpoint_pointers_are_role_and_kind_scoped() {
    let d = CheckpointDiscovery::default();
    let live_hash = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";

    // A trainer drain snapshot, a FRESHER trainer live checkpoint, and a coordinator drain
    // snapshot at an even higher round all publish (the node's CheckpointPublished hook target).
    RunDiscovery::publish_checkpoint(&d, "run-ckpt", "trainer", "drain", 7, "dd", 4096)
        .await
        .unwrap();
    RunDiscovery::publish_checkpoint(&d, "run-ckpt", "trainer", "live", 9, live_hash, 2048)
        .await
        .unwrap();
    RunDiscovery::publish_checkpoint(&d, "run-ckpt", "coordinator", "drain", 12, "cc", 512)
        .await
        .unwrap();

    // A trainer restores from ITS freshest LIVE pointer — the coordinator's higher-round drain
    // snapshot never shadows it (the per-role rule), and the drain slot never outranks a
    // fresher live one (the per-kind rule).
    let pointer = RunDiscovery::fetch_checkpoint(&d, "run-ckpt", "trainer")
        .await
        .unwrap()
        .expect("a trainer pointer is published");
    assert_eq!(pointer.kind, "live");
    assert_eq!(pointer.round, 9);
    assert_eq!(pointer.hash, live_hash);

    // With no live pointer, the role falls back to its own drain snapshot.
    let coord = RunDiscovery::fetch_checkpoint(&d, "run-ckpt", "coordinator")
        .await
        .unwrap()
        .expect("a coordinator pointer is published");
    assert_eq!(coord.kind, "drain");
    assert_eq!(coord.round, 12);

    // A role with no pointers starts fresh.
    assert!(RunDiscovery::fetch_checkpoint(&d, "run-ckpt", "verifier")
        .await
        .unwrap()
        .is_none());
}
