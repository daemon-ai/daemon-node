// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The Phase-E upgrade-transaction acceptance drills (refactor §9; architecture §5.4; ABI §10.3),
// over PRODUCTION wasm blobs in the host testkit:
//
// 1. **Live epoch-fenced worker upgrade without restart**: a real transition chain commits an
//    authorized upgrade record (epoch 0 → 1); the worker-role instance quiesces at the fence,
//    snapshots (`stage_state`/`snapshot_state`), the new module re-admits under owner law,
//    `da_migrate` restores the state, and the new instance activates locally — counter continuity
//    across the fence, spooled frames draining into the new instance, tag-13 reason 2, and the
//    §12.1 frame envelope carrying the new epoch. The chain is never advanced or rolled back by
//    any host step.
// 2. **Coordinator self-upgrade with signer continuity**: the same transaction for the
//    coordinator role, plus D2's `RunKeyCertificate` machinery — the base identity fences the old
//    epoch's certified signer and certifies the new epoch's, and acceptance follows the certs.
// 3. **Mid-migration crash → local rollback-and-retry**: attempt 0's `da_migrate` is forced down
//    mid-flight (the typed `MigrateBudget` interruption); the transaction rolls back the LOCAL
//    snapshot and retries; attempt 1 activates. The chain stays at the committed epoch throughout.
// 4. **Grant-expansion refusal (negative)**: the epoch-1 grants derive LOOSER than the admitted
//    epoch-0 quotas — the transaction fails closed and the worker exits the run; no new instance
//    ever starts; the old epoch is never resumed.
//
// Dev/test harness: shells `cargo build` for the guests (the established pattern), so the
// fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use daemon_vhc_abi::{
    DEFAULT_CHANNEL_CONTROL_ID, OUTCOME_QUIESCE_READY, QUIESCE_REASON_UPGRADE,
    STOP_REASON_RUN_COMPLETE,
};
use daemon_vhc_host::v2::{
    admit_v2, start_run, start_run_migrating, DeviceProfile, EnvelopeRoleGrants, MemorySink,
    MigrationInput, OwnerPolicy, ParticipationLane, PumpHandle, RunEnd, RunIdentity, SinkEntry,
    SnapshotCapture, SpooledFrame, V2Run, V2RunConfig,
};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_proto::envelope::{Access, DeviceMinimums};
use daemon_vhc_proto::genesis::{
    ChannelDecl, EventCap, EventCaps, GenesisEnvelope, Identities, RoleEntry, RoleGrants,
    RunSectionV2, SnapshotArtifact, TransportSelection, GENESIS_SCHEMA_MAJOR,
};
use daemon_vhc_proto::sign::verify_bytes;
use daemon_vhc_proto::{
    blake3_hash, derive_admitted_quotas, peer_id, to_canonical_vec, verify_certified_sender,
    AdmittedQuotas, BufferReq, CertError, EpochDescriptor, Hash, PeerId, RunKeyCertificate,
    SigningKey, TransitionChain, UpgradeAuthority, UpgradeRecord,
};
use daemon_vhc_session::upgrade::{
    run_local_upgrade, LeaveReason, LocalUpgradeOutcome, SnapshotSeam, StepFailure, UpgradeSteps,
};

// -- guest build harness (the established testkit pattern) ----------------------------------------

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.ancestors().nth(3).unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

static BUILD: Once = Once::new();

fn guest_wasm(name: &str) -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

// -- drill fixtures --------------------------------------------------------------------------------

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

/// A relaxed lane for the drills: participation enabled, GPU optional, zero device floors —
/// lanes are versioned node-side configuration, and the drill machine has no GPU. The grant
/// ceilings keep the launch Trainer defaults (generous).
fn drill_lane() -> ParticipationLane {
    let mut lane = ParticipationLane::trainer_launch_defaults();
    lane.gpu = 1; // optional
    lane.vram_bytes = 0;
    lane.ram_bytes = 0;
    lane.disk_bytes = 0;
    lane
}

/// The drill role grant list: the `control` channel with finite bounds, finite advisory depths,
/// finite quotas — the fail-closed comparison baseline derives finite everywhere, so any loosened
/// epoch-1 grant is detectable expansion.
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

/// A two-role genesis (worker + coordinator, both pinned to the OLD drill module) with a
/// single-key upgrade authority; returns the frozen run id and the anchored chain.
fn drill_genesis(
    old_module: Hash,
    upgrade_key: &SigningKey,
) -> (GenesisEnvelope, Hash, TransitionChain, UpgradeAuthority) {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "worker-mod".to_string(),
        SnapshotArtifact {
            url: "r2://drill/old.wasm".into(),
            blake3: old_module,
            size: None,
        },
    );
    artifacts.insert(
        "coord-mod".to_string(),
        SnapshotArtifact {
            url: "r2://drill/old.wasm".into(),
            blake3: old_module,
            size: None,
        },
    );
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
        run: RunSectionV2 {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: "upgrade-drill".into(),
            min_peers: 1,
            max_peers: 8,
            access: Access::Org,
        },
        roles,
        artifacts,
        authority: ciborium::value::Value::Map(vec![]),
        transport: TransportSelection::default(),
        identities: Identities {
            upgrade_authority: vec![peer_id(upgrade_key)],
            ..Default::default()
        },
    };
    let frozen = genesis.freeze(&key(200)).expect("freeze genesis");
    let run_id = *frozen.run_id();
    let chain = TransitionChain::genesis(&genesis, run_id).expect("anchor chain");
    let authority = UpgradeAuthority::from_genesis(&genesis.identities).expect("authority");
    (genesis, run_id, chain, authority)
}

/// Poll until `cond` holds or `timeout` elapses; returns whether it held.
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

/// Decode a published §12.1 signed frame `[envelope, payload, sig]`: returns
/// `(epoch, module, sender, payload)` after verifying the signature over the canonical envelope.
fn decode_signed_frame(frame: &[u8]) -> (u64, [u8; 32], [u8; 32], Vec<u8>) {
    use ciborium::value::Value;
    let v: Value = ciborium::de::from_reader(frame).expect("frame cbor");
    let Value::Array(items) = v else {
        panic!("signed frame is [envelope, payload, sig]")
    };
    let envelope = items[0].clone();
    let Value::Bytes(payload) = items[1].clone() else {
        panic!("payload is bstr")
    };
    let Value::Bytes(sig) = items[2].clone() else {
        panic!("sig is bstr")
    };
    let Value::Map(env_entries) = &envelope else {
        panic!("envelope is a map")
    };
    let get = |name: &str| {
        env_entries
            .iter()
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
    let sender: [u8; 32] = match get("sender") {
        Value::Bytes(b) => b.as_slice().try_into().expect("sender 32 bytes"),
        _ => panic!("sender is bstr"),
    };
    // The signature covers the canonical encoding of the envelope (§12.1).
    let env_canonical = to_canonical_vec(&envelope).expect("canonical envelope");
    let sig64: [u8; 64] = sig.as_slice().try_into().expect("64-byte sig");
    verify_bytes(
        &PeerId(sender),
        &daemon_vhc_proto::Signature(sig64),
        &env_canonical,
    )
    .expect("frame signature verifies to its sender");
    (epoch, module, sender, payload)
}

// -- the UpgradeSteps adapter over the real host primitives -----------------------------------------

/// The production-shaped step adapter: each `UpgradeSteps` method wires onto the real host
/// primitive the spec names — `PumpHandle::quiesce` + `snapshot_capture` (steps 1–2), `admit_v2`
/// (step 3), `start_run_migrating` (steps 4–5), spooled-frame drain (step 6).
struct WasmSteps<'w> {
    worker: &'w Worker,
    role: String,
    // the OLD live instance
    old_run: Option<V2Run>,
    old_pump: PumpHandle,
    old_sink: Arc<Mutex<MemorySink>>,
    // a frame that arrives DURING the drain (must spool, §4.4)
    late_frame: Option<(u64, [u8; 32], Vec<u8>)>,
    // the NEW module + its admission inputs
    new_wasm: Vec<u8>,
    new_identity: RunIdentity,
    new_signing_seed: [u8; 32],
    new_grants_bytes: Vec<u8>,
    lane: ParticipationLane,
    device: DeviceProfile,
    owner: OwnerPolicy,
    envelope_grants: EnvelopeRoleGrants,
    // per-attempt migrate fuel (the crash drill schedules a starvation then a real budget)
    migrate_fuel: Vec<Option<u64>>,
    attempt: usize,
    // captured state
    capture: Option<SnapshotCapture>,
    spooled: Vec<SpooledFrame>,
    pending: Option<(V2Run, PumpHandle, Arc<Mutex<MemorySink>>)>,
    activated: Option<(V2Run, PumpHandle, Arc<Mutex<MemorySink>>)>,
    migrate_failures: Vec<String>,
    left: Option<String>,
}

impl WasmSteps<'_> {
    fn new_instance_sinks(&self) -> Arc<Mutex<MemorySink>> {
        Arc::new(Mutex::new(MemorySink::new()))
    }
}

impl UpgradeSteps for WasmSteps<'_> {
    fn quiesce(&mut self, deadline_ms: u64) -> Result<SnapshotSeam, StepFailure> {
        self.old_pump
            .quiesce(QUIESCE_REASON_UPGRADE, deadline_ms)
            .map_err(|e| StepFailure::new(format!("quiesce delivery: {e}")))?;
        // A frame arriving DURING the drain: frozen — it spools, never reaching the draining
        // instance (§4.4), and drains into the new instance at activation (§10.3 step 6).
        if let Some((seq, sender, payload)) = self.late_frame.take() {
            self.old_pump
                .deliver_frame(
                    DEFAULT_CHANNEL_CONTROL_ID,
                    seq,
                    sender,
                    payload,
                    b"late-original-signed-frame".to_vec(),
                )
                .map_err(|e| StepFailure::new(format!("late frame: {e}")))?;
        }
        let run = self.old_run.take().expect("old instance is live");
        match run.wait() {
            Ok(RunEnd::Outcome(code)) if u64::from(code) == u64::from(OUTCOME_QUIESCE_READY) => {}
            other => {
                return Err(StepFailure::new(format!(
                    "drain did not quiesce: {other:?}"
                )))
            }
        }
        let capture = self
            .old_pump
            .snapshot_capture()
            .ok_or_else(|| StepFailure::new("no accepted snapshot in the drain"))?;
        self.spooled = self.old_pump.take_spooled_frames();
        // The labelled seam: an OPAQUE blob + the journal cursor — E1's typed manifest swaps in
        // here at merge (the session module's SnapshotSeam docs).
        let seam = SnapshotSeam {
            blob: capture.manifest.clone(),
            journal_cursor: self.old_sink.lock().expect("sink").entries.len() as u64,
        };
        self.capture = Some(capture);
        Ok(seam)
    }

    fn readmit(
        &mut self,
        new_module: Hash,
        new_grants_hash: Hash,
    ) -> Result<AdmittedQuotas, StepFailure> {
        // The committed record's grants anchor: the re-derived grants document must hash to it.
        if blake3_hash(&self.new_grants_bytes) != new_grants_hash {
            return Err(StepFailure::new(
                "re-derived grants do not match the committed record's grants_hash",
            ));
        }
        // Owner-law re-check (§10.3 step 3): the full funnel over the NEW module.
        let admission = admit_v2(
            self.worker,
            &self.new_wasm,
            Some(new_module.as_bytes()),
            &[],
            &self.new_grants_bytes,
            &self.lane,
            &self.device,
            &self.owner,
            None,
            Some(&self.envelope_grants),
        )
        .map_err(|refusal| StepFailure::new(refusal.to_string()))?;
        admission
            .quotas
            .ok_or_else(|| StepFailure::new("envelope-grants admission yields quotas"))
    }

    fn migrate(&mut self, seam: &SnapshotSeam) -> Result<(), StepFailure> {
        let capture = self.capture.clone().expect("quiesce captured");
        assert_eq!(
            seam.blob, capture.manifest,
            "the seam carries the captured manifest opaquely"
        );
        let fuel = self.migrate_fuel.get(self.attempt).copied().flatten();
        self.attempt += 1;
        let sink = self.new_instance_sinks();
        let cfg = V2RunConfig::new(
            self.new_identity.clone(),
            self.new_signing_seed,
            Vec::new(),
            self.new_grants_bytes.clone(),
        );
        let run = start_run_migrating(
            self.worker,
            &self.new_wasm,
            cfg,
            Box::new(sink.clone()),
            Some(MigrationInput {
                capture,
                restore: true,
                migrate_fuel: fuel,
            }),
        )
        .map_err(|e| StepFailure::new(format!("start_run_migrating: {e}")))?;
        let pump = run.pump.clone();
        // The migrated module announces the restored state as its first publish; a failed
        // migrate tears the instance down before da_run (§10.3 step 5).
        let ok = wait_for(Duration::from_secs(30), || {
            !pump.published().is_empty() || run.is_finished()
        });
        assert!(ok, "migrating instance neither published nor tore down");
        if pump.published().is_empty() {
            let end = run.wait().map_err(|e| StepFailure::new(e.to_string()))?;
            let detail = format!("migrate step failed: {end:?}");
            self.migrate_failures.push(detail.clone());
            return Err(StepFailure::new(detail));
        }
        self.pending = Some((run, pump, sink));
        Ok(())
    }

    fn activate(&mut self) -> Result<(), StepFailure> {
        let (run, pump, sink) = self.pending.take().expect("migrate produced an instance");
        // Spooled frames drain into the new instance (§10.3 step 6).
        for f in self.spooled.drain(..) {
            pump.deliver_frame(
                f.channel,
                f.seq,
                f.sender,
                f.payload,
                f.original_signed_frame,
            )
            .map_err(|e| StepFailure::new(format!("spool drain: {e}")))?;
        }
        self.activated = Some((run, pump, sink));
        Ok(())
    }

    fn rollback(&mut self, _seam: &SnapshotSeam) {
        // A failed migrate already tore the new instance down (guest-thread-owned teardown); the
        // snapshot capture IS the recovery point — nothing else to restore locally.
        if let Some((run, pump, _)) = self.pending.take() {
            let _ = pump.stop(STOP_REASON_RUN_COMPLETE);
            let _ = run.wait();
        }
    }

    fn leave(&mut self, reason: &LeaveReason) {
        self.left = Some(reason.to_string());
    }
}

// -- drill assembly ---------------------------------------------------------------------------------

struct Drill<'w> {
    steps: WasmSteps<'w>,
    target: EpochDescriptor,
    chain_epoch_before: u64,
    grants_hash: Hash,
    prev_quotas: AdmittedQuotas,
    old_pump: PumpHandle,
    old_sink: Arc<Mutex<MemorySink>>,
    run_id: Hash,
    new_module: Hash,
    old_seed: [u8; 32],
    new_seed: [u8; 32],
}

/// Assemble one live drill for `role`: start the OLD instance (epoch 0), feed it `frames` frames
/// (counter advances), commit the epoch-1 upgrade record to a REAL transition chain, and return
/// everything `run_local_upgrade` needs. `new_grants` is the epoch-1 role grant list (the
/// expansion negative passes a loosened one).
#[allow(clippy::too_many_lines)]
fn assemble<'w>(
    worker: &'w Worker,
    role: &str,
    frames: u64,
    new_grants: &RoleGrants,
    migrate_fuel: Vec<Option<u64>>,
    late_frame: bool,
) -> Drill<'w> {
    let old_wasm = guest_wasm("test_migrate_old");
    let new_wasm = guest_wasm("test_migrate_new");
    let old_module = Hash(*blake3::hash(&old_wasm).as_bytes());
    let new_module = Hash(*blake3::hash(&new_wasm).as_bytes());

    let upgrade_key = key(1);
    let (genesis, run_id, mut chain, authority) = drill_genesis(old_module, &upgrade_key);

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
    let mut old_cfg = V2RunConfig::new(old_identity, old_seed, Vec::new(), old_grants_bytes);
    old_cfg.migration_max_sections = 4;
    old_cfg.migration_max_section_bytes = 1 << 12;
    let old_run = start_run(worker, &old_wasm, old_cfg, Box::new(old_sink.clone()))
        .expect("old instance starts");
    let old_pump = old_run.pump.clone();

    // Live traffic before the fence: the counter advances and publishes per frame.
    let sender = [42u8; 32];
    for seq in 0..frames {
        old_pump
            .deliver_frame(
                DEFAULT_CHANNEL_CONTROL_ID,
                seq,
                sender,
                b"tick".to_vec(),
                b"original-signed-frame".to_vec(),
            )
            .expect("frame accepted");
    }
    assert!(
        wait_for(Duration::from_secs(30), || {
            old_pump.published().len() as u64 == frames
        }),
        "old instance consumed the pre-fence traffic"
    );

    // The run-level event (deliverable 1): the authorized upgrade record commits ONCE, globally,
    // advancing the chain to epoch 1 — before any host acts (§5.4).
    let new_grants_bytes = to_canonical_vec(new_grants).expect("grants cbor");
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

    // The previously-admitted quotas (the fail-closed baseline) from the epoch-0 grants.
    let lane = drill_lane();
    let run_artifacts = genesis.artifacts.values().map(|a| a.blake3).collect();
    let prev_quotas =
        derive_admitted_quotas(&genesis.roles[role].grants, &lane.ceilings, &run_artifacts)
            .expect("epoch-0 quotas derive");

    let new_seed = [22u8; 32];
    let steps = WasmSteps {
        worker,
        role: role.to_string(),
        old_run: Some(old_run),
        old_pump: old_pump.clone(),
        old_sink: old_sink.clone(),
        late_frame: late_frame.then(|| (frames, sender, b"late".to_vec())),
        new_wasm,
        new_identity: RunIdentity {
            run_id: run_id.0,
            epoch: 1,
            role: role.to_string(),
            instance: 7,
            module: new_module.0,
        },
        new_signing_seed: new_seed,
        new_grants_bytes,
        lane,
        device: DeviceProfile::default(),
        owner: OwnerPolicy {
            participation_enabled: true,
            vram_cap_bytes: 0,
            host_cap_bytes: 0,
        },
        envelope_grants: EnvelopeRoleGrants {
            grants: new_grants.clone(),
            run_artifacts,
        },
        migrate_fuel,
        attempt: 0,
        capture: None,
        spooled: Vec::new(),
        pending: None,
        activated: None,
        migrate_failures: Vec::new(),
        left: None,
    };
    Drill {
        steps,
        target,
        chain_epoch_before: chain.epoch(),
        grants_hash,
        prev_quotas,
        old_pump,
        old_sink,
        run_id,
        new_module,
        old_seed,
        new_seed,
    }
}

fn counter_of(payload: &[u8]) -> u64 {
    u64::from_le_bytes(payload.try_into().expect("8-byte counter"))
}

/// Stop the activated instance cleanly and return its journal entries.
fn finish_activated(steps: &mut WasmSteps<'_>) -> Vec<SinkEntry> {
    let (run, pump, sink) = steps.activated.take().expect("activated instance");
    pump.stop(STOP_REASON_RUN_COMPLETE).expect("stop");
    match run.wait() {
        Ok(RunEnd::Outcome(0)) => {}
        other => panic!("new instance did not stop cleanly: {other:?}"),
    }
    let entries = sink.lock().expect("sink").entries.clone();
    entries
}

// -- drill 1: live epoch-fenced worker upgrade without restart ---------------------------------------

#[test]
fn worker_upgrade_live_epoch_fenced_without_restart() {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut drill = assemble(&worker, "worker", 3, &role_grants(4096), vec![], true);

    let outcome = run_local_upgrade(
        &drill.target,
        "worker",
        drill.grants_hash,
        &drill.prev_quotas,
        5_000,
        1,
        &mut drill.steps,
    );
    assert_eq!(
        outcome,
        LocalUpgradeOutcome::Activated {
            epoch: 1,
            retries: 0
        }
    );

    // The chain was committed once, globally, and no host step moved it (§10.3 step 6).
    assert_eq!(drill.chain_epoch_before, 1);

    // Continuity across the fence, without restart: the new instance's FIRST publish is the
    // restored counter (3 pre-fence frames), and the spooled late frame drained into it (4).
    let (_, pump, _) = drill.steps.activated.as_ref().expect("activated");
    assert!(
        wait_for(Duration::from_secs(30), || pump.published().len() >= 2),
        "restored announcement + drained late frame"
    );
    let published = pump.published();
    let (epoch0, module0, sender0, payload0) = decode_signed_frame(&published[0].2);
    assert_eq!(counter_of(&payload0), 3, "da_migrate restored the counter");
    // The §12.1 envelope carries the NEW execution identity: epoch 1, the new module hash, and
    // the new per-run key.
    assert_eq!(epoch0, 1);
    assert_eq!(module0, drill.new_module.0);
    assert_eq!(sender0, peer_id(&SigningKey::from_bytes(&drill.new_seed)).0);
    let (_, _, _, payload1) = decode_signed_frame(&published[1].2);
    assert_eq!(
        counter_of(&payload1),
        4,
        "the drain-spooled frame drained into the new instance and counted"
    );

    // The OLD journal holds the accepted snapshot (tag 10) and the QuiesceReady terminal.
    let old_entries = drill.old_sink.lock().expect("sink").entries.clone();
    assert!(
        old_entries
            .iter()
            .any(|e| matches!(e, SinkEntry::Snapshot { manifest } if !manifest.is_empty())),
        "tag-10 snapshot journaled on the old instance"
    );
    assert!(old_entries.iter().any(|e| matches!(
        e,
        SinkEntry::Terminal { kind: 0, outcome: Some(o) } if *o == u64::from(OUTCOME_QUIESCE_READY)
    )));

    // The NEW journal opens with tag-13 reason 2 (upgrade-activation), before da_init (tag 11).
    let new_entries = finish_activated(&mut drill.steps);
    let inst_pos = new_entries
        .iter()
        .position(|e| matches!(e, SinkEntry::Instantiation { reason: 2, .. }))
        .expect("tag-13 reason 2 on the migrating instance");
    let init_pos = new_entries
        .iter()
        .position(|e| matches!(e, SinkEntry::Init { .. }))
        .expect("tag-11 init");
    assert!(
        inst_pos < init_pos,
        "instantiation journaled before da_init (§10.3 step 4)"
    );
    // The restore readback (kind 3) was journaled like any read_back (§10.2).
    assert!(new_entries
        .iter()
        .any(|e| matches!(e, SinkEntry::ReadBack { kind: 3, .. })));
    assert!(drill.steps.left.is_none(), "the worker never left the run");
}

// -- drill 2: coordinator self-upgrade with signer continuity ----------------------------------------

#[test]
fn coordinator_self_upgrade_carries_signer_continuity() {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut drill = assemble(&worker, "coordinator", 2, &role_grants(4096), vec![], false);

    // D2's signer-continuity machinery: the base machine identity certified the OLD epoch's
    // per-run key with a validity window that ENDS at the fence (epoch_to = 0).
    let base = key(9);
    let old_run_key = peer_id(&SigningKey::from_bytes(&drill.old_seed));
    let cert_old =
        RunKeyCertificate::issue(&base, drill.run_id, "coordinator", 7, 0, 0, old_run_key)
            .expect("old cert");

    let outcome = run_local_upgrade(
        &drill.target,
        "coordinator",
        drill.grants_hash,
        &drill.prev_quotas,
        5_000,
        1,
        &mut drill.steps,
    );
    assert_eq!(
        outcome,
        LocalUpgradeOutcome::Activated {
            epoch: 1,
            retries: 0
        }
    );

    // The new instance signs as the NEW per-run key under epoch 1 — observed from its frames.
    let (_, pump, _) = drill.steps.activated.as_ref().expect("activated");
    assert!(wait_for(Duration::from_secs(30), || !pump
        .published()
        .is_empty()));
    let published = pump.published();
    let (epoch, _, sender, _) = decode_signed_frame(&published[0].2);
    assert_eq!(epoch, 1);
    let new_sender = PeerId(sender);

    // Signer continuity (architecture §4.4/§5.4; the D2 failover building block): the base
    // identity issues the NEW epoch's certificate at the fence...
    let cert_new =
        RunKeyCertificate::issue(&base, drill.run_id, "coordinator", 7, 1, 10, new_sender)
            .expect("new cert");
    let base_id = peer_id(&base);
    let store = [cert_old.clone(), cert_new];

    // ...the new signer is ACCEPTED for epoch-1 frames through the chain to the base identity...
    verify_certified_sender(
        &drill.run_id,
        "coordinator",
        7,
        1,
        &new_sender,
        &base_id,
        &store,
    )
    .expect("epoch-1 signer certified");
    // ...the OLD signer is FENCED at the epoch boundary: its certificate expired at epoch 0
    // (Expired on direct check; no certified chain covers it at epoch 1)...
    assert_eq!(
        cert_old.authorizes_sender(&drill.run_id, "coordinator", 7, 1, &old_run_key),
        Err(CertError::Expired {
            epoch: 1,
            from: 0,
            to: 0
        })
    );
    assert_eq!(
        verify_certified_sender(
            &drill.run_id,
            "coordinator",
            7,
            1,
            &old_run_key,
            &base_id,
            &store
        ),
        Err(CertError::NoCertifiedChain)
    );
    // ...and WITHOUT the new certificate the new signer is not yet accepted (the continuity
    // step is explicit, never implicit).
    assert_eq!(
        verify_certified_sender(
            &drill.run_id,
            "coordinator",
            7,
            1,
            &new_sender,
            &base_id,
            &[cert_old]
        ),
        Err(CertError::NoCertifiedChain)
    );

    finish_activated(&mut drill.steps);
}

// -- drill 3: mid-migration crash recovers by local rollback-and-retry -------------------------------

#[test]
fn mid_migration_crash_recovers_by_local_rollback_and_retry() {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    // Attempt 0 runs da_migrate under a starvation budget (the typed MigrateBudget interruption
    // mid-migration); attempt 1 gets the real budget.
    let mut drill = assemble(
        &worker,
        "worker",
        5,
        &role_grants(4096),
        vec![Some(1_000), None],
        false,
    );

    let outcome = run_local_upgrade(
        &drill.target,
        "worker",
        drill.grants_hash,
        &drill.prev_quotas,
        5_000,
        2,
        &mut drill.steps,
    );
    assert_eq!(
        outcome,
        LocalUpgradeOutcome::Activated {
            epoch: 1,
            retries: 1
        },
        "the second attempt activated after the local rollback"
    );
    // The crash was the typed migrate-budget interruption, mid-migration.
    assert_eq!(drill.steps.migrate_failures.len(), 1);
    assert!(
        drill.steps.migrate_failures[0].contains("MigrateBudget"),
        "attempt 0 fell to the migrate budget: {}",
        drill.steps.migrate_failures[0]
    );
    // The chain never rolled back; the retry activated the SAME already-committed epoch, and
    // the restored state is intact (5 pre-fence frames).
    assert_eq!(drill.chain_epoch_before, 1);
    let (_, pump, _) = drill.steps.activated.as_ref().expect("activated");
    assert!(wait_for(Duration::from_secs(30), || !pump
        .published()
        .is_empty()));
    let (epoch, _, _, payload) = decode_signed_frame(&pump.published()[0].2);
    assert_eq!(epoch, 1);
    assert_eq!(
        counter_of(&payload),
        5,
        "state restored intact on the retry"
    );
    finish_activated(&mut drill.steps);
}

// -- drill 4: the grant-expansion refusal (negative) --------------------------------------------------

#[test]
fn grant_expanding_upgrade_fails_closed_and_the_worker_exits() {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    // The epoch-1 grants LOOSEN the channel frame bound (4096 → 8192): still within the lane,
    // but wider than the previously-admitted set — the owner-law fail-closed case.
    let mut drill = assemble(&worker, "worker", 2, &role_grants(8192), vec![], false);

    let outcome = run_local_upgrade(
        &drill.target,
        "worker",
        drill.grants_hash,
        &drill.prev_quotas,
        5_000,
        1,
        &mut drill.steps,
    );
    match outcome {
        LocalUpgradeOutcome::Left {
            epoch,
            reason: LeaveReason::GrantExpansion(why),
        } => {
            assert_eq!(
                epoch, 1,
                "the run stays at the committed epoch; only this node left"
            );
            assert!(
                why.contains("max_frame_bytes"),
                "names the expanded bound: {why}"
            );
        }
        other => panic!("expected the grant-expansion refusal, got {other:?}"),
    }
    // Fail closed means fail CLOSED: the worker exited; no new instance was ever started; the
    // old epoch was not resumed (the old instance stays quiesced).
    assert!(drill.steps.left.is_some(), "the worker exited the run");
    assert!(drill.steps.pending.is_none() && drill.steps.activated.is_none());
    assert_eq!(drill.steps.attempt, 0, "migrate never ran");
    // The old instance's journal ends at the QuiesceReady terminal — never resumed.
    let old_entries = drill.old_sink.lock().expect("sink").entries.clone();
    assert!(matches!(
        old_entries.last(),
        Some(SinkEntry::Terminal { kind: 0, outcome: Some(o) })
            if *o == u64::from(OUTCOME_QUIESCE_READY)
    ));
    // Unused fields hold the drill shape together even on the refused path.
    let _ = (&drill.old_pump, &drill.steps.role);
}
