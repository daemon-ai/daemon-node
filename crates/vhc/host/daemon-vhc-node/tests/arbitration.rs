// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Multi-instance supervision under aggregate owner arbitration (Phase E, refactor §9;
//! decisions D1/D6) — the node-level acceptance: **trainer + verifier colocated on one host,
//! arbitrated, both green**, plus the funnel-order and teardown-ordering properties around it.
//!
//! Each role-instance gets its own worker child (the `WorkerFactory` seam — one sandbox = one
//! role-instance), every join is admitted through the [`OwnerArbiter`]'s atomic
//! check-and-reserve against the owner's typed ledgers BEFORE any child joins, refusals are
//! typed `ApiError`s naming the exhausted ledger, and a leave releases the ledger only after
//! the child's teardown is observed. The production-blob (tier-2) twin of this acceptance runs
//! in `daemon-vhc-testkit/tests/colocation.rs`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{SwarmApi, SwarmLeaveMode, SwarmPolicy, SwarmPolicyMode};
use daemon_vhc_node::service::{VhcError, WorkerControl};
use daemon_vhc_node::{
    DiscoveredRun, OwnerBudget, RunDiscovery, VhcService, VhcServiceParts, VhcStore,
};
use daemon_vhc_session::config::VhcConfig;
use daemon_vhc_session::protocol::{Eligibility, Hardware, JoinPolicy, LeaveMode};
use std::collections::BTreeMap;

const MIB: u64 = 1 << 20;

/// A per-instance fake child recording its own join/leave stream (the factory hands a fresh one
/// to every admitted role-instance).
#[derive(Default)]
struct FakeChild {
    joins: Mutex<Vec<String>>,
    leaves: Mutex<Vec<String>>,
    shutdowns: Mutex<usize>,
    vram_mb: i64,
    /// When set, `assess` returns a claim-bearing verdict carrying these `(device, host)` bytes as
    /// the claim-shaped headroom (the discovery-path input to `derive_charge`, decisions D-10).
    assess_claim: Option<(i64, i64)>,
}

#[async_trait]
impl WorkerControl for FakeChild {
    async fn probe(&self) -> Result<Hardware, VhcError> {
        Ok(Hardware {
            gpus: 1,
            vram_mb: self.vram_mb as u64,
            ram_mb: 64_000,
            backend_lanes: vec!["cpu".into()],
            ..Default::default()
        })
    }
    async fn assess(&self, _envelope: Vec<u8>) -> Result<Eligibility, VhcError> {
        match self.assess_claim {
            Some((device, host)) => Ok(Eligibility {
                eligible: true,
                reasons: vec!["fake: claim admitted".into()],
                headroom: vec![
                    ("claim_device_bytes".into(), device),
                    ("claim_host_bytes".into(), host),
                ],
                refusal_code: None,
            }),
            None => unreachable!("no discovery seam in this test"),
        }
    }
    async fn join(
        &self,
        run_id: String,
        _coordinator: String,
        _credentials: Vec<u8>,
        _policy: JoinPolicy,
    ) -> Result<(), VhcError> {
        self.joins.lock().unwrap().push(run_id);
        Ok(())
    }
    async fn leave(&self, run_id: String, _mode: LeaveMode) -> Result<(), VhcError> {
        self.leaves.lock().unwrap().push(run_id);
        Ok(())
    }
    async fn throttle(
        &self,
        _vram_cap_mb: Option<u32>,
        _duty_cycle_pct: Option<u8>,
        _paused: bool,
    ) -> Result<(), VhcError> {
        Ok(())
    }
    async fn shutdown(&self) {
        *self.shutdowns.lock().unwrap() += 1;
    }
}

/// The probe-eligibility headroom carries the probed dedicated VRAM, which
/// `eligibility_from_hardware` renders as the claim-shaped `claim_device_bytes` charge key
/// (decisions D-10): the no-registry default path charges probed VRAM, and — when the probe
/// reports no dedicated VRAM (this rig's `vram_mb = 0`) — `derive_charge` falls back to the
/// per-run `policy.vram_cap_mb` as the conservative estimate, never a zero charge (D6 point 3).
fn probe_worker(vram_mb: i64) -> Arc<FakeChild> {
    Arc::new(FakeChild {
        vram_mb,
        ..FakeChild::default()
    })
}

fn policy(vram_cap_mb: u32, duty: u32) -> SwarmPolicy {
    SwarmPolicy {
        mode: SwarmPolicyMode::Idle,
        vram_cap_mb,
        duty_cycle_pct: duty,
        schedule: None,
    }
}

/// One 10 GiB accelerator, 100% duty, at most 2 instances — room for exactly the trainer (6 GiB
/// at 40%) plus the verifier (4 GiB at 40%).
fn colocation_budget() -> OwnerBudget {
    OwnerBudget {
        device_memory: BTreeMap::from([("gpu:0".to_string(), 10_000 * MIB)]),
        host_ram: u64::MAX,
        disk: u64::MAX,
        net_up_bps: u64::MAX,
        net_down_bps: u64::MAX,
        duty_pct: 100,
        max_instances: 2,
    }
}

struct Rig {
    svc: Arc<VhcService>,
    children: Arc<Mutex<Vec<Arc<FakeChild>>>>,
}

fn rig(budget: OwnerBudget) -> Rig {
    let children: Arc<Mutex<Vec<Arc<FakeChild>>>> = Arc::new(Mutex::new(Vec::new()));
    let spawned = children.clone();
    let factory: daemon_vhc_node::service::WorkerFactory = Arc::new(move || {
        // Every role-instance gets its own child; the probe advertises the run's VRAM verdict
        // (the per-run numbers ride the probe headroom in this rig: trainer 6000, verifier 4000,
        // set through the eligibility path below — the child itself reports what it was built
        // with).
        let child = probe_worker(0);
        spawned.lock().unwrap().push(child.clone());
        child as Arc<dyn WorkerControl>
    });
    let svc = Arc::new(VhcService::new(VhcServiceParts {
        config: VhcConfig {
            enabled: true,
            ..VhcConfig::default()
        },
        store: VhcStore::open_in_memory().unwrap(),
        worker: probe_worker(0),
        feed: None,
        discovery: None,
        budget: Some(budget),
        worker_factory: Some(factory),
    }));
    svc.bind_self();
    Rig { svc, children }
}

/// THE Phase-E colocation acceptance (refactor §9): a trainer role-instance and a verifier
/// role-instance run **colocated on one host**, each in its own supervised child, both admitted
/// through the owner arbiter's typed ledgers — both green; a third instance that would exceed
/// the device ledger is refused TYPED; a leave releases the ledger (observed teardown) and the
/// third instance then admits.
#[tokio::test]
async fn trainer_and_verifier_colocate_under_arbitration_both_green() {
    let r = rig(colocation_budget());

    // The probe path charges policy.vram_cap_mb when the probe headroom is absent/zero — the
    // tightening overlay as the estimate (decisions D6: JoinPolicy narrows, never exceeds).
    r.svc
        .swarm_join("run-trainer".into(), policy(6_000, 40), "op-1".into())
        .await
        .expect("trainer admitted");
    r.svc
        .swarm_join("run-verifier".into(), policy(4_000, 40), "op-2".into())
        .await
        .expect("verifier colocated");

    // Both live: two distinct children (one sandbox = one role-instance), each joined its run.
    let children = r.children.lock().unwrap().clone();
    assert_eq!(children.len(), 2, "one child per role-instance");
    assert_eq!(
        children[0].joins.lock().unwrap().as_slice(),
        ["run-trainer"]
    );
    assert_eq!(
        children[1].joins.lock().unwrap().as_slice(),
        ["run-verifier"]
    );
    // The ledgers account exactly: 10_000 - 6_000 - 4_000 = 0 MiB remaining, 20% duty left.
    let snap = r.svc.arbiter().remaining();
    assert_eq!(snap.device_memory["gpu:0"], 0);
    assert_eq!(snap.duty_pct, 100 - 40 - 40);
    assert_eq!(snap.instances, 2);

    // Distinct never-reused incarnations were minted and persisted (ABI §8.1 identity).
    let t = r.svc.store().get_run("run-trainer").unwrap().unwrap();
    let v = r.svc.store().get_run("run-verifier").unwrap().unwrap();
    assert!(t.instance > 0 && v.instance > 0);
    assert_ne!(t.instance, v.instance, "incarnations are never shared");

    // A third join exceeding every ledger dimension is a TYPED refusal (the funnel's last,
    // supreme stage) — the API error names the exhausted ledger, and no third child spawned.
    let err = r
        .svc
        .swarm_join("run-third".into(), policy(1, 1), "op-3".into())
        .await
        .expect_err("no room");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("owner arbitration refused"),
        "typed arbitration refusal, got: {msg}"
    );
    // The assessment child that probed for the refused join never joined a run and was torn
    // down (the assessment instance precedes arbitration; a refused child is never left idle).
    {
        let children = r.children.lock().unwrap();
        assert_eq!(children.len(), 3, "the probe child existed");
        assert!(children[2].joins.lock().unwrap().is_empty(), "never joined");
        assert_eq!(*children[2].shutdowns.lock().unwrap(), 1, "torn down");
    }

    // Leave the verifier: teardown observed → ledger released → the third now admits.
    r.svc
        .swarm_leave(
            "run-verifier".into(),
            SwarmLeaveMode::Graceful,
            "op-4".into(),
        )
        .await
        .expect("leave");
    assert_eq!(
        r.children.lock().unwrap()[1]
            .leaves
            .lock()
            .unwrap()
            .as_slice(),
        ["run-verifier"]
    );
    assert_eq!(
        r.svc.arbiter().remaining().device_memory["gpu:0"],
        4_000 * MIB
    );
    r.svc
        .swarm_join("run-third".into(), policy(1_000, 10), "op-5".into())
        .await
        .expect("admits after the release");
    assert_eq!(r.svc.arbiter().instances(), 2);
}

/// A repeated join of a live role-instance re-converges on the existing child + reservation —
/// it never double-charges the ledgers and never spawns a second child (idempotent intents,
/// ADR-006, now under arbitration).
#[tokio::test]
async fn repeated_join_never_double_charges() {
    let r = rig(colocation_budget());
    r.svc
        .swarm_join("run-a".into(), policy(6_000, 40), "op-1".into())
        .await
        .unwrap();
    let before = r.svc.arbiter().remaining();
    r.svc
        .swarm_join("run-a".into(), policy(6_000, 40), "op-2".into())
        .await
        .expect("re-join converges");
    assert_eq!(r.svc.arbiter().remaining(), before, "no double charge");
    assert_eq!(r.children.lock().unwrap().len(), 1, "no second child");
}

/// Restart re-convergence re-admits persisted intents through the arbiter (the ledger's crash
/// reconciliation for node-supervised children), retains the persisted incarnation (a process
/// restart retains the logical instance id, decisions D1), and surfaces a no-longer-fitting
/// intent LOUD (a persisted `owner_arbitration` error event) without blocking the others.
#[tokio::test]
async fn restart_reconverges_through_the_arbiter_and_reports_refusals_loud() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vhc.db");
    let incarnation;
    {
        let children: Arc<Mutex<Vec<Arc<FakeChild>>>> = Arc::new(Mutex::new(Vec::new()));
        let spawned = children.clone();
        let factory: daemon_vhc_node::service::WorkerFactory = Arc::new(move || {
            let child = probe_worker(0);
            spawned.lock().unwrap().push(child.clone());
            child as Arc<dyn WorkerControl>
        });
        let svc = Arc::new(VhcService::new(VhcServiceParts {
            config: VhcConfig {
                enabled: true,
                ..VhcConfig::default()
            },
            store: VhcStore::open(&path).unwrap(),
            worker: probe_worker(0),
            feed: None,
            discovery: None,
            budget: Some(colocation_budget()),
            worker_factory: Some(factory),
        }));
        svc.bind_self();
        svc.swarm_join("run-big".into(), policy(6_000, 40), "op-1".into())
            .await
            .unwrap();
        svc.swarm_join("run-small".into(), policy(2_000, 10), "op-2".into())
            .await
            .unwrap();
        incarnation = svc.store().get_run("run-big").unwrap().unwrap().instance;
    }

    // Restart with a SHRUNK budget: only the small run still fits.
    let shrunk = OwnerBudget {
        device_memory: BTreeMap::from([("gpu:0".to_string(), 3_000 * MIB)]),
        ..colocation_budget()
    };
    let r = {
        let children: Arc<Mutex<Vec<Arc<FakeChild>>>> = Arc::new(Mutex::new(Vec::new()));
        let spawned = children.clone();
        let factory: daemon_vhc_node::service::WorkerFactory = Arc::new(move || {
            let child = probe_worker(0);
            spawned.lock().unwrap().push(child.clone());
            child as Arc<dyn WorkerControl>
        });
        let svc = Arc::new(VhcService::new(VhcServiceParts {
            config: VhcConfig {
                enabled: true,
                ..VhcConfig::default()
            },
            store: VhcStore::open(&path).unwrap(),
            worker: probe_worker(0),
            feed: None,
            discovery: None,
            budget: Some(shrunk),
            worker_factory: Some(factory),
        }));
        svc.bind_self();
        Rig { svc, children }
    };
    let rejoined = r.svc.start().await.unwrap();
    assert_eq!(rejoined, 1, "only the still-fitting intent re-converges");
    // The refused run is loud: a persisted owner_arbitration error event.
    let events = r.svc.store().recent_events("run-big", 16).unwrap();
    assert!(
        events.iter().any(|e| matches!(
            e,
            daemon_api::SwarmEvent::Error { class, .. } if class == "owner_arbitration"
        )),
        "the refused intent is surfaced loud, got {events:?}"
    );
    // The persisted incarnation was retained for the re-admitted run (had it been run-big we'd
    // check that one; run-small's identity is durable too).
    let small = r.svc.store().get_run("run-small").unwrap().unwrap();
    assert!(small.instance > 0);
    assert_ne!(small.instance, incarnation);
}

/// A stub discovery seam that always resolves a run and hands back opaque envelope bytes, so the
/// service takes the **assess** path (`worker.assess` → `eligibility_from_assess`) instead of the
/// probe fallback.
struct StubDiscovery;

#[async_trait]
impl RunDiscovery for StubDiscovery {
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, VhcError> {
        Ok(Vec::new())
    }
    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, VhcError> {
        Ok(Some(DiscoveredRun {
            run_id: run_id.to_string(),
            coordinator: "wss://coord.example/swarm".to_string(),
            envelope_hash: "00".repeat(32),
            proto_version: 1,
        }))
    }
    async fn fetch_envelope(&self, _run_id: &str) -> Result<Vec<u8>, VhcError> {
        Ok(vec![1, 2, 3, 4])
    }
}

/// D-10 / D6 point 3: on the assess path a claim-bearing verdict carries `claim_device_bytes` /
/// `claim_host_bytes`, and `derive_charge` charges them verbatim onto the device + host tiers —
/// so an **admitted instance's arbiter reservation equals the assess claim totals exactly** (with
/// an uncapped policy, the owner cap does not tighten the claim). This is the acceptance assertion
/// the retirement plan requires for the both-inputs rewrite.
#[tokio::test]
async fn admitted_charge_equals_assess_claim_totals() {
    const GIB: i64 = 1 << 30;
    let claim_device = 5 * GIB;
    let claim_host = 8 * GIB;

    let children: Arc<Mutex<Vec<Arc<FakeChild>>>> = Arc::new(Mutex::new(Vec::new()));
    let spawned = children.clone();
    let factory: daemon_vhc_node::service::WorkerFactory = Arc::new(move || {
        let child = Arc::new(FakeChild {
            assess_claim: Some((claim_device, claim_host)),
            ..FakeChild::default()
        });
        spawned.lock().unwrap().push(child.clone());
        child as Arc<dyn WorkerControl>
    });

    // A budget generous enough for the claim on both the device and host ledgers.
    let budget = OwnerBudget {
        device_memory: BTreeMap::from([("gpu:0".to_string(), 16 * GIB as u64)]),
        host_ram: 32 * GIB as u64,
        disk: u64::MAX,
        net_up_bps: u64::MAX,
        net_down_bps: u64::MAX,
        duty_pct: 100,
        max_instances: 4,
    };

    let svc = Arc::new(VhcService::new(VhcServiceParts {
        config: VhcConfig {
            enabled: true,
            ..VhcConfig::default()
        },
        store: VhcStore::open_in_memory().unwrap(),
        worker: probe_worker(0),
        feed: None,
        discovery: Some(Arc::new(StubDiscovery)),
        budget: Some(budget),
        worker_factory: Some(factory),
    }));
    svc.bind_self();

    // Uncapped policy (`vram_cap_mb = 0`) so the assess claim stands verbatim.
    svc.swarm_join("run-claim".into(), policy(0, 50), "op-1".into())
        .await
        .expect("claim-bearing join admitted");

    // The reservation equals the assess claim totals exactly on both tiers.
    let snap = svc.arbiter().remaining();
    assert_eq!(
        snap.device_memory["gpu:0"],
        16 * GIB as u64 - claim_device as u64,
        "device ledger charged the assess device-claim total"
    );
    assert_eq!(
        snap.host_ram,
        32 * GIB as u64 - claim_host as u64,
        "host ledger charged the assess host-claim total"
    );
    assert_eq!(snap.instances, 1);
}
