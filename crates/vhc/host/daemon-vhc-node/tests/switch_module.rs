// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// Node-level acceptance for the LIVE module-upgrade command surface (ABI §10.3; architecture
// §5.4). This is NOT a testkit drill: it drives `WorkerControl::switch_module` — the node's worker
// command seam — end to end against a REAL running worker instance (a real wasm migratable guest
// under the host runtime), through the production upgrade transaction
// (`daemon_vhc_session::upgrade::{run_local_upgrade, LiveUpgradeSteps}`), and asserts:
//
//   - the old module quiesces at the fence and snapshots its state,
//   - the new module re-admits under owner law, migrates the state, and resumes — without a
//     process restart, its first publish being the restored (continuous) det state,
//   - the §12.1 frame envelope carries the NEW execution identity (the target epoch + module),
//   - the transition is recorded per the transition-chain / epoch rules (the chain advanced once,
//     globally, to the target epoch; no host step moved it), and
//   - the node command surface reports [`SwitchOutcome::Activated`].
//
// Dev/test harness: shells `cargo build` for the guests (the established testkit pattern).
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command as ProcCommand;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use daemon_vhc_abi::{DEFAULT_CHANNEL_CONTROL_ID, STOP_REASON_RUN_COMPLETE};
use daemon_vhc_host::run::{
    start_run, DeviceProfile, EnvelopeRoleGrants, MemorySink, OwnerPolicy, ParticipationLane,
    PumpHandle, RunConfig, RunEnd, RunIdentity,
};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_node::{VhcError, WorkerControl};
use daemon_vhc_proto::envelope::{Access, DeviceMinimums};
use daemon_vhc_proto::genesis::{
    ChannelDecl, EventCap, EventCaps, GenesisEnvelope, Identities, RoleEntry, RoleGrants,
    RunSection, SnapshotArtifact, TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::{
    blake3_hash, derive_admitted_quotas, peer_id, to_canonical_vec, AdmittedQuotas, BufferReq,
    EpochDescriptor, Hash, SigningKey, TransitionChain, UpgradeAuthority, UpgradeRecord,
};
use daemon_vhc_session::protocol::{Eligibility, Hardware, JoinPolicy, LeaveMode};
use daemon_vhc_session::upgrade::{
    run_local_upgrade, ActivatedInstance, LiveUpgradeInputs, LiveUpgradeSteps, LocalUpgradeOutcome,
};
use daemon_vhc_supervisor::SwitchOutcome;

// -- guest build harness (the established testkit pattern) ----------------------------------------

fn guests_root() -> PathBuf {
    // crates/vhc/host/daemon-vhc-node -> crates/vhc/guests
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

static BUILD: Once = Once::new();

fn guest_wasm(name: &str) -> Vec<u8> {
    BUILD.call_once(|| {
        let status = ProcCommand::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn wait_for(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    cond()
}

fn counter_of(payload: &[u8]) -> u64 {
    u64::from_le_bytes(payload.try_into().expect("8-byte counter"))
}

/// Decode a published §12.1 signed frame `[envelope, payload, sig]` → `(epoch, module, payload)`.
fn decode_signed_frame(frame: &[u8]) -> (u64, [u8; 32], Vec<u8>) {
    use ciborium::value::Value;
    let v: Value = ciborium::de::from_reader(frame).expect("frame cbor");
    let Value::Array(items) = v else {
        panic!("signed frame is [envelope, payload, sig]")
    };
    let Value::Bytes(payload) = items[1].clone() else {
        panic!("payload is bstr")
    };
    let Value::Map(env) = &items[0] else {
        panic!("envelope is a map")
    };
    let get = |name: &str| {
        env.iter()
            .find_map(|(k, val)| match k {
                Value::Text(t) if t == name => Some(val.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("envelope field {name}"))
    };
    let epoch = get("epoch")
        .as_integer()
        .map(|i| u64::try_from(i128::from(i)).unwrap())
        .expect("epoch uint");
    let module: [u8; 32] = match get("module") {
        Value::Bytes(b) => b.as_slice().try_into().expect("module 32 bytes"),
        _ => panic!("module is bstr"),
    };
    (epoch, module, payload)
}

// -- the scenario the node command surface acts on ------------------------------------------------

/// The role grant list (finite bounds everywhere, so the fail-closed baseline is detectable).
fn role_grants(max_frame_bytes: u64) -> RoleGrants {
    RoleGrants {
        channels: vec![ChannelDecl {
            id: DEFAULT_CHANNEL_CONTROL_ID,
            name: "control".into(),
            class: 0,
            direction: 2,
            max_frame_bytes,
            rate_per_min: 600,
            spool_frames: Some(64),
            replay_window: Some(128),
            per_sender_quota: Some(16),
        }],
        events: EventCaps {
            classes: [
                (
                    "timer".to_string(),
                    EventCap {
                        depth: 8,
                        coalesce: 1,
                    },
                ),
                (
                    "payload-ready".to_string(),
                    EventCap {
                        depth: 8,
                        coalesce: 0,
                    },
                ),
                (
                    "gossip".to_string(),
                    EventCap {
                        depth: 8,
                        coalesce: 2,
                    },
                ),
            ]
            .into_iter()
            .collect(),
        },
        buffers: BufferReq {
            max_live_handles: 8,
            max_live_bytes: 1 << 16,
            max_readback_bytes: 1 << 16,
        },
        max_outstanding_ops: 4,
        compute_queue_depth: 8,
        ..Default::default()
    }
}

fn drill_lane() -> ParticipationLane {
    let mut lane = ParticipationLane::trainer_launch_defaults();
    lane.gpu = 1;
    lane.vram_bytes = 0;
    lane.ram_bytes = 0;
    lane.disk_bytes = 0;
    lane
}

/// Everything the node's `switch_module` seam needs to run the LOCAL upgrade transaction against
/// the live OLD instance (built once, consumed by the switch).
struct Scenario {
    role: String,
    old_run_inputs: LiveScenarioInputs,
    target: EpochDescriptor,
    grants_hash: Hash,
    prev_quotas: AdmittedQuotas,
    max_retries: u32,
}

/// The owned pieces of the live OLD instance + NEW module admission bundle (moved into
/// `LiveUpgradeInputs` at switch time; a struct so the borrow of `&Worker` happens only there).
struct LiveScenarioInputs {
    old_run: daemon_vhc_host::run::Run,
    old_pump: PumpHandle,
    old_sink: Arc<Mutex<MemorySink>>,
    old_module: Hash,
    new_wasm: Vec<u8>,
    new_identity: RunIdentity,
    new_signing_seed: [u8; 32],
    new_grants_bytes: Vec<u8>,
    lane: ParticipationLane,
    device: DeviceProfile,
    owner: OwnerPolicy,
    envelope_grants: EnvelopeRoleGrants,
    drain_window_frame: Option<(u64, [u8; 32], Vec<u8>)>,
}

/// Build the live scenario: start the OLD migratable guest (epoch 0), feed it `frames` frames, and
/// commit an authorized epoch-1 upgrade record to a REAL transition chain. Returns the scenario
/// plus the committed chain (for the transition-recording assertion).
fn build_scenario(worker: &Worker, frames: u64) -> (Scenario, TransitionChain) {
    let old_wasm = guest_wasm("test_migrate_old");
    let new_wasm = guest_wasm("test_migrate_new");
    let old_module = Hash(*blake3::hash(&old_wasm).as_bytes());
    let new_module = Hash(*blake3::hash(&new_wasm).as_bytes());
    let role = "worker";

    // A single-role genesis pinned to the OLD module, with a single-key upgrade authority.
    let upgrade_key = key(1);
    let mut artifacts = BTreeMap::new();
    for name in ["worker-mod", "coord-mod"] {
        artifacts.insert(
            name.to_string(),
            SnapshotArtifact {
                url: "r2://acceptance/old.wasm".into(),
                blake3: old_module,
                size: None,
            },
        );
    }
    let mut roles = BTreeMap::new();
    for (name, module, lane) in [
        ("worker", "worker-mod", "trainer"),
        ("coordinator", "coord-mod", "coordinator"),
    ] {
        roles.insert(
            name.to_string(),
            RoleEntry {
                lane: lane.into(),
                module: module.into(),
                abi: "vhc@2".into(),
                config: ciborium::value::Value::Map(vec![]),
                grants: role_grants(4096),
                device_min: DeviceMinimums::default(),
            },
        );
    }
    let genesis = GenesisEnvelope {
        run: RunSection {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: "switch-module-acceptance".into(),
            min_peers: 1,
            max_peers: 8,
            access: Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: None,
        authority: ciborium::value::Value::Map(vec![]),
        transport: TransportSelection::default(),
        identities: Identities {
            upgrade_authority: vec![peer_id(&upgrade_key)],
            ..Default::default()
        },
    };
    let frozen = genesis.freeze(&key(200)).expect("freeze genesis");
    let run_id = *frozen.run_id();
    let mut chain = TransitionChain::genesis(&genesis, run_id).expect("anchor chain");
    let authority = UpgradeAuthority::from_genesis(&genesis.identities).expect("authority");

    // The OLD instance at epoch 0 — the real event-loop driver, journaled.
    let old_seed = [11u8; 32];
    let old_sink = Arc::new(Mutex::new(MemorySink::new()));
    let old_identity = RunIdentity {
        run_id: run_id.0,
        epoch: 0,
        role: role.to_string(),
        instance: 7,
        module: old_module.0,
    };
    let old_grants_bytes = to_canonical_vec(&genesis.roles[role].grants).expect("grants cbor");
    let mut old_cfg = RunConfig::new(old_identity, old_seed, Vec::new(), old_grants_bytes);
    old_cfg.migration_max_sections = 4;
    old_cfg.migration_max_section_bytes = 1 << 12;
    let old_run =
        start_run(worker, &old_wasm, old_cfg, Box::new(old_sink.clone())).expect("old instance");
    let old_pump = old_run.pump.clone();

    // Live traffic before the fence: the counter advances per frame.
    let sender = [42u8; 32];
    for seq in 0..frames {
        old_pump
            .deliver_frame(
                DEFAULT_CHANNEL_CONTROL_ID,
                seq,
                sender,
                b"tick".to_vec(),
                b"pre-fence-signed-frame".to_vec(),
            )
            .expect("frame accepted");
    }
    assert!(
        wait_for(Duration::from_secs(30), || old_pump.published().len()
            as u64
            == frames),
        "old instance consumed the pre-fence traffic"
    );

    // The run-level event (deliverable 1): the authorized upgrade record commits ONCE, globally,
    // advancing the chain to epoch 1 — before any host acts (§5.4).
    let new_grants = role_grants(4096);
    let new_grants_bytes = to_canonical_vec(&new_grants).expect("grants cbor");
    let grants_hash = blake3_hash(&new_grants_bytes);
    let record = UpgradeRecord::author(
        run_id,
        1,
        run_id,
        role,
        old_module,
        new_module,
        7,
        grants_hash,
        blake3_hash(&[]),
        &[&upgrade_key],
    )
    .expect("author record");
    let target = chain.append(record, &authority).expect("global commit");
    assert_eq!(target.epoch, 1);
    assert_eq!(target.module_for(role), Some(new_module));

    let lane = drill_lane();
    let run_artifacts = genesis.artifacts.values().map(|a| a.blake3).collect();
    let prev_quotas =
        derive_admitted_quotas(&genesis.roles[role].grants, &lane.ceilings, &run_artifacts)
            .expect("epoch-0 quotas derive");

    let scenario = Scenario {
        role: role.to_string(),
        old_run_inputs: LiveScenarioInputs {
            old_run,
            old_pump,
            old_sink,
            old_module,
            new_wasm,
            new_identity: RunIdentity {
                run_id: run_id.0,
                epoch: 1,
                role: role.to_string(),
                instance: 7,
                module: new_module.0,
            },
            new_signing_seed: [22u8; 32],
            new_grants_bytes,
            lane,
            device: DeviceProfile::default(),
            owner: OwnerPolicy {
                participation_enabled: true,
                vram_cap_bytes: 0,
                host_cap_bytes: 0,
            },
            envelope_grants: EnvelopeRoleGrants {
                grants: new_grants,
                run_artifacts,
            },
            // A frame arriving during the drain spools and drains into the new instance (§10.3
            // step 6) — proves continuity of the authoritative stream across the fence.
            drain_window_frame: Some((frames, sender, b"drain-window".to_vec())),
        },
        target,
        grants_hash,
        prev_quotas,
        max_retries: 1,
    };
    (scenario, chain)
}

// -- the in-process worker the node command surface drives ----------------------------------------

/// A [`WorkerControl`] that holds a REAL live role-instance and implements `switch_module` by
/// running the production upgrade transaction over it — the node's worker seam, exercised without
/// a subprocess (the transaction is the same `run_local_upgrade` + `LiveUpgradeSteps` the worker
/// binary would drive).
struct InProcessUpgradeWorker {
    worker: Worker,
    scenario: Mutex<Option<Scenario>>,
    activated: Mutex<Option<ActivatedInstance>>,
}

impl InProcessUpgradeWorker {
    fn new(worker: Worker, scenario: Scenario) -> Self {
        Self {
            worker,
            scenario: Mutex::new(Some(scenario)),
            activated: Mutex::new(None),
        }
    }
}

#[async_trait]
impl WorkerControl for InProcessUpgradeWorker {
    async fn probe(&self) -> Result<Hardware, VhcError> {
        Ok(Hardware::default())
    }
    async fn assess(&self, _envelope: Vec<u8>) -> Result<Eligibility, VhcError> {
        Ok(Eligibility {
            eligible: true,
            ..Eligibility::default()
        })
    }
    async fn join(
        &self,
        _run_id: String,
        _coordinator: String,
        _credentials: Vec<u8>,
        _policy: JoinPolicy,
    ) -> Result<(), VhcError> {
        Ok(())
    }
    async fn leave(&self, _run_id: String, _mode: LeaveMode) -> Result<(), VhcError> {
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

    async fn switch_module(
        &self,
        _run_id: String,
        epoch: u64,
        role: String,
        new_module: [u8; 32],
        grants_hash: [u8; 32],
        deadline_ms: u64,
    ) -> Result<SwitchOutcome, VhcError> {
        let scenario = self
            .scenario
            .lock()
            .expect("scenario")
            .take()
            .ok_or_else(|| VhcError::Worker("no live role-instance to switch".into()))?;
        // Hash-pinned + authority-bound: the command's target must match the committed chain's.
        if scenario.target.epoch != epoch
            || scenario.role != role
            || scenario.target.module_for(&role) != Some(Hash(new_module))
            || scenario.grants_hash != Hash(grants_hash)
        {
            return Err(VhcError::Worker(
                "SwitchModule target does not match the committed transition-chain descriptor"
                    .into(),
            ));
        }
        let inputs = scenario.old_run_inputs;
        let mut steps = LiveUpgradeSteps::new(LiveUpgradeInputs {
            worker: &self.worker,
            role: scenario.role.clone(),
            old_run: inputs.old_run,
            old_pump: inputs.old_pump,
            old_sink: inputs.old_sink,
            old_module: inputs.old_module,
            new_wasm: inputs.new_wasm,
            new_identity: inputs.new_identity,
            new_signing_seed: inputs.new_signing_seed,
            new_grants_bytes: inputs.new_grants_bytes,
            lane: inputs.lane,
            device: inputs.device,
            owner: inputs.owner,
            envelope_grants: inputs.envelope_grants,
            migrate_fuel: vec![],
            drain_window_frame: inputs.drain_window_frame,
        });
        let outcome = run_local_upgrade(
            &scenario.target,
            &role,
            Hash(grants_hash),
            &scenario.prev_quotas,
            deadline_ms,
            scenario.max_retries,
            &mut steps,
        );
        match outcome {
            LocalUpgradeOutcome::Activated { epoch, retries } => {
                let activated = steps.take_activated().expect("activated instance");
                *self.activated.lock().expect("activated") = Some(activated);
                Ok(SwitchOutcome::Activated {
                    epoch,
                    module: new_module,
                    retries,
                })
            }
            LocalUpgradeOutcome::Left { reason, .. } => Ok(SwitchOutcome::Left {
                reason: reason.to_string(),
            }),
        }
    }
}

// -- the acceptance -------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_switch_module_upgrades_a_running_instance_epoch_fenced() {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let (scenario, chain) = build_scenario(&worker, 3);
    // The committed chain advanced ONCE to epoch 1 (deliverable 1); no host step moves it.
    assert_eq!(chain.epoch(), 1);
    let target = scenario.target.clone();
    let new_module = target.module_for("worker").expect("target module");
    let grants_hash = scenario.grants_hash;

    let control = InProcessUpgradeWorker::new(worker, scenario);

    // Drive the NODE command surface: `WorkerControl::switch_module` (deadline clamped by the
    // node's `[vhc.upgrade].quiesce_deadline_max_ms` in production; here 5s).
    let outcome = WorkerControl::switch_module(
        &control,
        "switch-module-acceptance".to_string(),
        1,
        "worker".to_string(),
        new_module.0,
        grants_hash.0,
        5_000,
    )
    .await
    .expect("switch_module");

    assert_eq!(
        outcome,
        SwitchOutcome::Activated {
            epoch: 1,
            module: new_module.0,
            retries: 0
        },
        "the node command surface reports the activation of the target epoch"
    );

    // The old module quiesced, migrated, and the new module resumed — continuity across the fence:
    // its first publish is the restored counter (3 pre-fence frames), under the NEW epoch/module.
    let guard = control.activated.lock().expect("activated");
    let activated = guard.as_ref().expect("activated instance held");
    assert!(
        wait_for(Duration::from_secs(30), || activated.pump.published().len()
            >= 2),
        "restored announcement + drained drain-window frame"
    );
    let published = activated.pump.published();
    let (epoch0, module0, payload0) = decode_signed_frame(&published[0].2);
    assert_eq!(
        counter_of(&payload0),
        3,
        "the new module restored the state"
    );
    assert_eq!(epoch0, 1, "the §12.1 envelope carries the target epoch");
    assert_eq!(module0, new_module.0, "and the new module hash");
    let (_, _, payload1) = decode_signed_frame(&published[1].2);
    assert_eq!(
        counter_of(&payload1),
        4,
        "the drain-window frame drained into the new instance and counted (continuity)"
    );

    // The transition is recorded per the chain/epoch rules: the run globally sits at epoch 1 with
    // the new module bound to the role — advanced by the committed record, not by any host step.
    assert_eq!(chain.epoch(), 1);
    assert_eq!(chain.module_for("worker"), Some(new_module));

    // Clean teardown of the activated instance (guest-thread-owned).
    drop(guard);
    let activated = control.activated.lock().expect("activated").take().unwrap();
    activated.pump.stop(STOP_REASON_RUN_COMPLETE).expect("stop");
    match activated.run.wait() {
        Ok(RunEnd::Outcome(0)) => {}
        other => panic!("new instance did not stop cleanly: {other:?}"),
    }
}
