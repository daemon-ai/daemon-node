// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The live module switch under the role session (ABI §10.3), over the pinned migrate drill
//! pair: the FROM module counts delivered frames and snapshots its counter at the drain fence;
//! the TO module restores the counter through `da_migrate` and announces it as its first
//! publish — state continuity across the epoch fence, observable as opaque payload bytes.
//!
//! Covered here: the happy path (state continuity + one continued journal + publish sequences
//! restarting at 0 in the new incarnation's stream + the re-issued certificate signing
//! post-switch frames), the pre-fence refusals (tampered target artifact, grant expansion,
//! mis-scoped re-issued certificate — the old module keeps running untouched through each), and
//! a post-fence migrate failure (starved migrate budget → bounded retry → the session leaves the
//! run typed).
//!
//! Dev/test harness: shells `cargo build` for the guests (the established pattern), so the
//! fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::Duration;

use daemon_vhc_host::run::{
    DeviceProfile, EnvelopeRoleGrants, MemorySink, OwnerPolicy, ParticipationLane, RunConfig,
    RunIdentity, SinkEntry,
};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_net::{ControlPlane, LoopbackGossip, MemoryContentStore};
use daemon_vhc_proto::genesis::{ChannelDecl, RoleGrants};
use daemon_vhc_proto::{peer_id, AdmittedQuotas, CertScope, Hash, SigningKey};
use daemon_vhc_session::distribution::DistributionRecord;
use daemon_vhc_session::identity::{issue_run_key, CertifiedRunKey};
use daemon_vhc_session::protocol::{AdmittedTuple, Event, LeaveMode, TerminalOutcome};
use daemon_vhc_session::role_session::{spawn_role, RoleProviders, RoleSessionSpec, SwitchBinding};
use tokio::sync::mpsc;

// -- guest build harness (mirrors the sibling suites) ----------------------------------------------

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
            .env_remove("RUSTC_WRAPPER")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

// -- rig --------------------------------------------------------------------------------------------

const RUN_ID: [u8; 32] = [0x77u8; 32];
const ROLE: &str = "counter";

/// The role's grant list: the control channel both drill modules publish on.
fn role_grants() -> RoleGrants {
    RoleGrants {
        channels: vec![ChannelDecl {
            id: 0,
            name: "control".into(),
            class: 0,
            direction: 2,
            max_frame_bytes: 1 << 20,
            rate_per_min: 600,
            spool_frames: Some(256),
            replay_window: Some(1024),
            per_sender_quota: Some(64),
        }],
        ..RoleGrants::default()
    }
}

fn lane() -> ParticipationLane {
    ParticipationLane {
        gpu: 1,
        vram_bytes: 0,
        ram_bytes: 0,
        disk_bytes: 0,
        ..ParticipationLane::trainer_launch_defaults()
    }
}

fn device() -> DeviceProfile {
    DeviceProfile {
        gpu: true,
        vram_bytes: 8 << 30,
        ram_bytes: 16 << 30,
        disk_bytes: 100 << 30,
    }
}

fn owner() -> OwnerPolicy {
    OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    }
}

struct Rig {
    plane: Arc<LoopbackGossip>,
    sink: Arc<Mutex<MemorySink>>,
    spec: RoleSessionSpec,
    base: SigningKey,
}

/// A session over the FROM drill module (config byte 0 = the counter's start), joined at
/// epoch 0 / incarnation 1 with in-process providers and a shared memory journal.
fn rig(wasm: &[u8], admitted_quotas: Option<AdmittedQuotas>) -> Rig {
    let module_hash = *blake3::hash(wasm).as_bytes();
    let identity = RunIdentity {
        run_id: RUN_ID,
        epoch: 0,
        role: ROLE.to_string(),
        instance: 1,
        module: module_hash,
    };
    let base = daemon_vhc_session::identity::SecretSeed::fresh()
        .expect("entropy")
        .signing_key();
    let certified: CertifiedRunKey = issue_run_key(
        &base,
        CertScope {
            run_id: Hash(RUN_ID),
            epoch: 0,
            role: ROLE.to_string(),
            instance: 1,
            module_hash: Hash(module_hash),
        },
    )
    .expect("issue run key");
    let run_cfg = RunConfig::new(
        identity,
        certified.key.to_bytes(),
        vec![0u8],
        b"admitted-grants".to_vec(),
    );
    let plane = Arc::new(LoopbackGossip::new());
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let spec = RoleSessionSpec {
        module: wasm.to_vec(),
        engine: EngineConfig::default(),
        run: run_cfg,
        own_cert: certified.cert.clone(),
        trusted_bases: vec![peer_id(&base)],
        peer_certs: vec![certified.cert],
        providers: RoleProviders {
            control: plane.clone(),
            payloads: Arc::new(MemoryContentStore::new()),
            artifacts: Arc::new(MemoryContentStore::new()),
        },
        journal: Box::new(sink.clone()),
        drain_deadline: Duration::from_secs(10),
        restore: None,
        admitted_quotas,
    };
    Rig {
        plane,
        sink,
        spec,
        base,
    }
}

/// Author one §12.1 wire frame signed by `key` (the same envelope shape the pump signs).
#[allow(clippy::too_many_arguments)]
fn signed_frame(
    key: &CertifiedRunKey,
    epoch: u64,
    module_hash: [u8; 32],
    role: &str,
    instance: u64,
    seq: u64,
    payload: &[u8],
) -> Vec<u8> {
    use ciborium::value::Value;
    let sender = key.sender().0;
    let envelope = Value::Map(vec![
        (Value::from("domain"), Value::from("daemon-vhc/frame/2")),
        (Value::from("run_id"), Value::Bytes(RUN_ID.to_vec())),
        (Value::from("epoch"), Value::from(epoch)),
        (Value::from("role"), Value::from(role)),
        (Value::from("instance"), Value::from(instance)),
        (Value::from("module"), Value::Bytes(module_hash.to_vec())),
        (Value::from("sender"), Value::Bytes(sender.to_vec())),
        (Value::from("channel"), Value::from(0u64)),
        (Value::from("seq"), Value::from(seq)),
        (
            Value::from("payload_hash"),
            Value::Bytes(blake3::hash(payload).as_bytes().to_vec()),
        ),
    ]);
    let sig = daemon_vhc_proto::sign::sign_canonical(&key.key, &envelope).expect("sign");
    let wire = Value::Array(vec![
        envelope,
        Value::Bytes(payload.to_vec()),
        Value::Bytes(sig.0.to_vec()),
    ]);
    let mut out = Vec::new();
    ciborium::into_writer(&wire, &mut out).expect("frame cbor");
    out
}

/// The switch binding a well-behaved node would author for `new_wasm` at `epoch`/`incarnation`:
/// grants re-derived for the target under the run's role grants, the admission claim
/// re-evaluated for the tuple, the identity re-issued under `base`. The seam journal continues
/// `old_sink`'s entries; the opened continuation lands in `seam_slot` for assertions.
#[allow(clippy::too_many_arguments)]
fn switch_binding_for(
    new_wasm: &[u8],
    base: &SigningKey,
    epoch: u64,
    incarnation: u64,
    old_sink: &Arc<Mutex<MemorySink>>,
    seam_slot: &Arc<Mutex<Option<Arc<Mutex<MemorySink>>>>>,
    migrate_fuel: Option<u64>,
) -> (SwitchBinding, CertifiedRunKey) {
    let new_module = *blake3::hash(new_wasm).as_bytes();
    let admission_worker = Worker::new(EngineConfig::default()).expect("admission engine");
    let linked =
        daemon_vhc_host::linked_worlds(&admission_worker, new_wasm).expect("linked worlds");
    let grants = daemon_vhc_proto::GrantsDoc::author(&linked, &role_grants()).to_canonical_bytes();
    let grants_hash = *blake3::hash(&grants).as_bytes();
    let envelope_grants = EnvelopeRoleGrants {
        grants: role_grants(),
        run_artifacts: std::collections::BTreeSet::new(),
    };
    let admission = daemon_vhc_host::run::admit(
        &admission_worker,
        new_wasm,
        Some(&new_module),
        &[],
        &grants,
        &lane(),
        &device(),
        &owner(),
        None,
        Some(&envelope_grants),
    )
    .expect("target admits");
    let tuple = AdmittedTuple {
        module_hash: new_module,
        config_hash: *blake3::hash(&[]).as_bytes(),
        grants_hash,
        claim_hash: *blake3::hash(&admission.claim_bytes).as_bytes(),
        genesis_hash: RUN_ID,
        role: ROLE.to_string(),
        incarnation,
        device_profile_rev: 0,
        owner_policy_rev: 0,
        backend: "cpu".to_string(),
        gpu_index: 0,
    };
    let certified = issue_run_key(
        base,
        CertScope {
            run_id: Hash(RUN_ID),
            epoch,
            role: ROLE.to_string(),
            instance: incarnation,
            module_hash: Hash(new_module),
        },
    )
    .expect("re-issue run key");
    let journal: daemon_vhc_session::role_session::SeamJournal = {
        let old_sink = old_sink.clone();
        let seam_slot = seam_slot.clone();
        Box::new(move |_id: &RunIdentity| {
            // The logical seam over the memory sink: the retired incarnation's records remain
            // as the prefix; the continuation resets the publish high-water marks (§12.2).
            let prefix = old_sink.lock().expect("old sink").entries.clone();
            let cont = Arc::new(Mutex::new(MemorySink::continuing(prefix)));
            *seam_slot.lock().expect("seam slot") = Some(cont.clone());
            Ok(Box::new(cont) as Box<dyn daemon_vhc_host::run::JournalSink>)
        })
    };
    let binding = SwitchBinding {
        epoch,
        new_module,
        grants_hash,
        tuple,
        module_bytes: Some(new_wasm.to_vec()),
        config: Vec::new(),
        signing_seed: certified.key.to_bytes(),
        own_cert: certified.cert.clone(),
        role_grants: role_grants(),
        envelope_grants: Some(envelope_grants),
        lane: lane(),
        device: device(),
        owner: owner(),
        journal,
        deadline_ms: 10_000,
        migrate_fuel,
    };
    (binding, certified)
}

/// Await the next event `pick` accepts, skipping metrics/warnings chatter.
async fn until<T>(
    events: &mut mpsc::UnboundedReceiver<Event>,
    mut pick: impl FnMut(&Event) -> Option<T>,
) -> T {
    loop {
        let ev = tokio::time::timeout(Duration::from_secs(60), events.recv())
            .await
            .expect("event within the deadline")
            .expect("event stream open");
        if let Some(out) = pick(&ev) {
            return out;
        }
    }
}

/// Payloads published on the loopback plane, decoded from the §12.1 wire triple (the TEST reads
/// the opaque payload; the session never does).
fn frame_payload(frame: &[u8]) -> Option<(Vec<u8>, [u8; 32], u64, u64)> {
    use ciborium::value::Value;
    let v: Value = ciborium::de::from_reader(frame).ok()?;
    let Value::Array(parts) = v else { return None };
    let Value::Map(env) = parts.first()? else {
        return None;
    };
    let field = |name: &str| {
        env.iter().find_map(|(k, v)| match k {
            Value::Text(t) if t == name => Some(v.clone()),
            _ => None,
        })
    };
    let Some(Value::Bytes(sender)) = field("sender") else {
        return None;
    };
    let epoch = field("epoch")?.as_integer()?;
    let seq = field("seq")?.as_integer()?;
    let Some(Value::Bytes(payload)) = parts.get(1).cloned() else {
        return None;
    };
    Some((
        payload,
        sender.as_slice().try_into().ok()?,
        u64::try_from(i128::from(epoch)).ok()?,
        u64::try_from(i128::from(seq)).ok()?,
    ))
}

/// Feed one certified peer frame and await the module's next counter publish on the plane.
async fn feed_and_read_counter(
    plane: &Arc<LoopbackGossip>,
    outside: &mut daemon_vhc_net::ControlSubscription,
    peer: &CertifiedRunKey,
    epoch: u64,
    module_hash: [u8; 32],
    seq: u64,
    session_sender: [u8; 32],
) -> u64 {
    let frame = signed_frame(peer, epoch, module_hash, "feeder", 1, seq, b"tick");
    plane.publish(&frame).await.expect("publish peer frame");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let bytes = tokio::time::timeout_at(deadline, outside.recv())
            .await
            .expect("counter publish within the deadline")
            .expect("plane open");
        if let Some((payload, sender, _, _)) = frame_payload(&bytes) {
            if sender == session_sender && payload.len() == 8 {
                return u64::from_le_bytes(payload.try_into().expect("8-byte counter"));
            }
        }
    }
}

/// Happy path: quiesce → snapshot → migrate → validate → activate. The restored counter proves
/// state continuity; the continued journal holds the retired span as its prefix; the new
/// incarnation's publishes restart at seq 0 under the re-issued certificate and the new epoch.
#[tokio::test(flavor = "multi_thread")]
async fn switch_migrates_state_continues_journal_and_restarts_publish_seq() {
    let old_wasm = guest_wasm("test_migrate_old");
    let new_wasm = guest_wasm("test_migrate_new");
    let r = rig(&old_wasm, None);
    let old_module = *blake3::hash(&old_wasm).as_bytes();
    let new_module = *blake3::hash(&new_wasm).as_bytes();
    let plane = r.plane.clone();
    let mut outside = plane.subscribe();
    let base = r.base;

    // A certified feeder peer at epoch 0 (its frames drive the counter).
    let feeder0 = issue_run_key(
        &base,
        CertScope {
            run_id: Hash(RUN_ID),
            epoch: 0,
            role: "feeder".into(),
            instance: 1,
            module_hash: Hash(old_module),
        },
    )
    .expect("feeder key");
    let mut spec = r.spec;
    spec.peer_certs.push(feeder0.cert.clone());
    let session_sender0 = spec.own_cert.body.run_key.0;

    let (tx, mut events) = mpsc::unbounded_channel();
    let handle = spawn_role("run-switch".into(), spec, tx);
    assert_eq!(handle.generation(), 1);
    // The session subscribes to the plane before it reports the running phase: feed only after.
    until(&mut events, |ev| match ev {
        Event::RunPhase { phase, .. } if phase == "running" => Some(()),
        _ => None,
    })
    .await;

    // Drive the counter to 3 under the OLD module.
    for want in 1..=3u64 {
        let got = feed_and_read_counter(
            &plane,
            &mut outside,
            &feeder0,
            0,
            old_module,
            want - 1,
            session_sender0,
        )
        .await;
        assert_eq!(got, want, "the FROM module counts delivered frames");
    }

    // The node-authored switch: epoch 1, incarnation 2, re-issued identity.
    let seam_slot: Arc<Mutex<Option<Arc<Mutex<MemorySink>>>>> = Arc::new(Mutex::new(None));
    let (binding, new_key) = switch_binding_for(&new_wasm, &base, 1, 2, &r.sink, &seam_slot, None);
    let new_sender = new_key.sender().0;
    handle.switch(binding);

    let (epoch, module, retries, generation) = until(&mut events, |ev| match ev {
        Event::ModuleSwitched {
            epoch,
            module,
            retries,
            generation,
            ..
        } => Some((*epoch, *module, *retries, *generation)),
        Event::SwitchRefused { reason, .. } => panic!("switch refused: {reason}"),
        Event::RunTerminated { outcome, .. } => panic!("switch left the run: {outcome:?}"),
        _ => None,
    })
    .await;
    assert_eq!(epoch, 1);
    assert_eq!(module, new_module);
    assert_eq!(retries, 0);
    assert_eq!(generation, 2, "the switch minted the new incarnation");
    assert_eq!(handle.generation(), 2, "the handle's generation advanced");

    // State continuity: the TO module announces the RESTORED counter (3) as its first publish —
    // signed by the RE-ISSUED key, under the NEW epoch, at seq 0 (the fresh §12.2 stream).
    let (payload, _sender, frame_epoch, seq) = loop {
        let bytes = tokio::time::timeout(Duration::from_secs(60), outside.recv())
            .await
            .expect("announce within the deadline")
            .expect("plane open");
        if let Some(decoded) = frame_payload(&bytes) {
            if decoded.1 == new_sender {
                break decoded;
            }
        }
    };
    assert_eq!(
        u64::from_le_bytes(payload.try_into().expect("8-byte counter")),
        3,
        "the migrated module restored the drained counter"
    );
    assert_eq!(frame_epoch, 1, "post-switch frames carry the new epoch");
    assert_eq!(seq, 0, "the publish sequence restarted in the new stream");

    // Journal continuity: the continued sink holds the retired span (its run-header + the three
    // counter publishes, seqs 0..=2) as the prefix, then the NEW span's run-header and the
    // announce publish at seq 0.
    let seam = seam_slot
        .lock()
        .unwrap()
        .clone()
        .expect("the seam journal opened");
    {
        let entries = &seam.lock().unwrap().entries;
        let headers = entries
            .iter()
            .filter(|e| matches!(e, SinkEntry::RunHeader { .. }))
            .count();
        assert_eq!(
            headers, 2,
            "one continued log, two execution-identity spans"
        );
        let publish_seqs: Vec<u64> = entries
            .iter()
            .filter_map(|e| match e {
                SinkEntry::Publish { seq, .. } => Some(*seq),
                _ => None,
            })
            .collect();
        assert_eq!(
            publish_seqs,
            vec![0, 1, 2, 0],
            "the retired stream's seqs stay as the prefix; the new stream restarts at 0"
        );
    }

    // The run CONTINUES: a certified epoch-1 feeder (its certificate distributed on the plane)
    // drives the restored counter forward.
    let feeder1 = issue_run_key(
        &base,
        CertScope {
            run_id: Hash(RUN_ID),
            epoch: 1,
            role: "feeder".into(),
            instance: 2,
            module_hash: Hash(new_module),
        },
    )
    .expect("feeder epoch-1 key");
    let record = DistributionRecord::Cert(feeder1.cert.clone())
        .to_bytes()
        .expect("cert record");
    plane.publish(&record).await.expect("distribute cert");
    let frame = signed_frame(&feeder1, 1, new_module, "feeder", 2, 0, b"tick");
    plane.publish(&frame).await.expect("publish epoch-1 frame");
    let counter = loop {
        let bytes = tokio::time::timeout(Duration::from_secs(60), outside.recv())
            .await
            .expect("post-switch counter within the deadline")
            .expect("plane open");
        if let Some((payload, sender, _, _)) = frame_payload(&bytes) {
            if sender == new_sender && payload.len() == 8 {
                let v = u64::from_le_bytes(payload.try_into().expect("8 bytes"));
                if v > 3 {
                    break v;
                }
            }
        }
    };
    assert_eq!(counter, 4, "the migrated module keeps counting");

    handle.leave(LeaveMode::Immediate);
    let outcome = until(&mut events, |ev| match ev {
        Event::RunTerminated {
            generation,
            outcome,
            ..
        } => Some((*generation, outcome.clone())),
        _ => None,
    })
    .await;
    assert_eq!(
        outcome,
        (2, TerminalOutcome::Left { checkpoint: None }),
        "the terminal event carries the post-switch generation"
    );
}

/// A tampered target artifact refuses BEFORE the fence: typed `SwitchRefused`, and the old
/// module keeps running (it still counts delivered frames afterwards).
#[tokio::test(flavor = "multi_thread")]
async fn tampered_target_artifact_refuses_with_the_old_module_unharmed() {
    let old_wasm = guest_wasm("test_migrate_old");
    let new_wasm = guest_wasm("test_migrate_new");
    let r = rig(&old_wasm, None);
    let old_module = *blake3::hash(&old_wasm).as_bytes();
    let plane = r.plane.clone();
    let mut outside = plane.subscribe();
    let base = r.base;
    let feeder = issue_run_key(
        &base,
        CertScope {
            run_id: Hash(RUN_ID),
            epoch: 0,
            role: "feeder".into(),
            instance: 1,
            module_hash: Hash(old_module),
        },
    )
    .expect("feeder key");
    let mut spec = r.spec;
    spec.peer_certs.push(feeder.cert.clone());
    let session_sender = spec.own_cert.body.run_key.0;
    let (tx, mut events) = mpsc::unbounded_channel();
    let handle = spawn_role("run-tamper".into(), spec, tx);
    // The session subscribes to the plane before it reports the running phase: feed only after.
    until(&mut events, |ev| match ev {
        Event::RunPhase { phase, .. } if phase == "running" => Some(()),
        _ => None,
    })
    .await;

    let seam_slot = Arc::new(Mutex::new(None));
    let (mut binding, _) = switch_binding_for(&new_wasm, &base, 1, 2, &r.sink, &seam_slot, None);
    // The delivered bytes do not hash to the committed target: refused at the hash pin.
    binding.module_bytes = Some(b"not-the-committed-artifact".to_vec());
    handle.switch(binding);

    let reason = until(&mut events, |ev| match ev {
        Event::SwitchRefused { reason, .. } => Some(reason.clone()),
        Event::ModuleSwitched { .. } => panic!("a tampered artifact must never activate"),
        Event::RunTerminated { outcome, .. } => panic!("refusal must not leave: {outcome:?}"),
        _ => None,
    })
    .await;
    assert!(reason.contains("does not hash"), "typed refusal: {reason}");
    assert_eq!(handle.generation(), 1, "no incarnation advance on refusal");

    // The old module is unharmed: it still counts.
    let got = feed_and_read_counter(
        &plane,
        &mut outside,
        &feeder,
        0,
        old_module,
        0,
        session_sender,
    )
    .await;
    assert_eq!(got, 1, "the running instance kept its state and its seat");

    handle.leave(LeaveMode::Immediate);
    until(&mut events, |ev| {
        matches!(ev, Event::RunTerminated { .. }).then_some(())
    })
    .await;
}

/// A grant-expanding upgrade refuses fail-closed before the fence (the re-admitted quotas are
/// compared against the quotas the instance was admitted under).
#[tokio::test(flavor = "multi_thread")]
async fn grant_expanding_switch_refuses_fail_closed() {
    let old_wasm = guest_wasm("test_migrate_old");
    let new_wasm = guest_wasm("test_migrate_new");
    // The join's baseline is TIGHTER than what the switch would re-admit (its channel grants a
    // 1 MiB frame ceiling): expansion → fail closed.
    let tight = AdmittedQuotas {
        max_frame_bytes: 1024,
        ..AdmittedQuotas::default()
    };
    let r = rig(&old_wasm, Some(tight));
    let base = r.base;
    let (tx, mut events) = mpsc::unbounded_channel();
    let handle = spawn_role("run-expand".into(), r.spec, tx);

    let seam_slot = Arc::new(Mutex::new(None));
    let (binding, _) = switch_binding_for(&new_wasm, &base, 1, 2, &r.sink, &seam_slot, None);
    handle.switch(binding);

    let reason = until(&mut events, |ev| match ev {
        Event::SwitchRefused { reason, .. } => Some(reason.clone()),
        Event::ModuleSwitched { .. } => panic!("a grant-expanding switch must never activate"),
        Event::RunTerminated { outcome, .. } => panic!("refusal must not leave: {outcome:?}"),
        _ => None,
    })
    .await;
    assert!(
        reason.contains("grant") && reason.contains("expand"),
        "typed fail-closed refusal: {reason}"
    );

    handle.leave(LeaveMode::Immediate);
    until(&mut events, |ev| {
        matches!(ev, Event::RunTerminated { .. }).then_some(())
    })
    .await;
}

/// A re-issued certificate that binds a different execution identity than the switch refuses
/// before the fence (the identity handshake fails closed; the old module keeps running).
#[tokio::test(flavor = "multi_thread")]
async fn mis_scoped_reissued_certificate_refuses_before_the_fence() {
    let old_wasm = guest_wasm("test_migrate_old");
    let new_wasm = guest_wasm("test_migrate_new");
    let r = rig(&old_wasm, None);
    let base = r.base;
    let (tx, mut events) = mpsc::unbounded_channel();
    let handle = spawn_role("run-cert".into(), r.spec, tx);

    let seam_slot = Arc::new(Mutex::new(None));
    let (mut binding, _) = switch_binding_for(&new_wasm, &base, 1, 2, &r.sink, &seam_slot, None);
    // Re-issue against the WRONG module hash: the certificate no longer binds the post-switch
    // execution identity.
    let wrong = issue_run_key(
        &base,
        CertScope {
            run_id: Hash(RUN_ID),
            epoch: 1,
            role: ROLE.to_string(),
            instance: 2,
            module_hash: Hash([0xEE; 32]),
        },
    )
    .expect("mis-scoped key");
    binding.own_cert = wrong.cert;
    binding.signing_seed = wrong.key.to_bytes();
    handle.switch(binding);

    let reason = until(&mut events, |ev| match ev {
        Event::SwitchRefused { reason, .. } => Some(reason.clone()),
        Event::ModuleSwitched { .. } => panic!("a mis-scoped certificate must never activate"),
        Event::RunTerminated { outcome, .. } => panic!("refusal must not leave: {outcome:?}"),
        _ => None,
    })
    .await;
    assert!(
        reason.contains("different execution identity"),
        "typed refusal: {reason}"
    );
    assert_eq!(handle.generation(), 1);

    handle.leave(LeaveMode::Immediate);
    until(&mut events, |ev| {
        matches!(ev, Event::RunTerminated { .. }).then_some(())
    })
    .await;
}

/// A migrate failure past the fence: the starved migrate budget traps every attempt, the bounded
/// rollback-and-retry exhausts, and the session LEAVES the run typed (the old epoch is never
/// resumed — coherent terminal state, ABI §10.3 step 7).
#[tokio::test(flavor = "multi_thread")]
async fn migrate_budget_exhaustion_leaves_the_run_typed() {
    let old_wasm = guest_wasm("test_migrate_old");
    let new_wasm = guest_wasm("test_migrate_new");
    let r = rig(&old_wasm, None);
    let base = r.base;
    let (tx, mut events) = mpsc::unbounded_channel();
    let handle = spawn_role("run-starve".into(), r.spec, tx);

    let seam_slot = Arc::new(Mutex::new(None));
    // One unit of migrate fuel: da_migrate traps MigrateBudget on every attempt.
    let (binding, _) = switch_binding_for(&new_wasm, &base, 1, 2, &r.sink, &seam_slot, Some(1));
    handle.switch(binding);

    let mut saw_retry = false;
    let (generation, outcome) = until(&mut events, |ev| match ev {
        Event::Warning { class, .. } if class == "switch_retry" => {
            saw_retry = true;
            None
        }
        Event::RunTerminated {
            generation,
            outcome,
            ..
        } => Some((*generation, outcome.clone())),
        Event::ModuleSwitched { .. } => panic!("a starved migrate must never activate"),
        _ => None,
    })
    .await;
    assert!(saw_retry, "the transaction rolled back and retried first");
    assert!(
        matches!(
            &outcome,
            TerminalOutcome::FailedTerminal { reason }
                if reason.contains("exhausted its retry budget")
        ),
        "typed terminal after the bounded retries: {outcome:?}"
    );
    // The terminal is stamped with the generation that was still live (the switch never
    // activated, so no incarnation advance happened).
    assert_eq!(generation, 1);
    assert!(
        tokio::time::timeout(Duration::from_secs(10), handle.join())
            .await
            .is_ok(),
        "the session task ended coherently"
    );
}
