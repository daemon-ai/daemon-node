// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The Phase-5 crash-restart regression — the PRODUCT reconstruction path** (§8.8; plan
// "coordinator-reconstruction"): where `failover.rs` proves the ALGORITHM over the dev-side
// oracle journal (`daemon-vhc-observe` records, `AttestedHead`s, testkit chain recovery), this
// drill proves the PRODUCTION machinery end to end:
//
// 1. A primary `coordinator_quorum.wasm` drives rounds under the LIVE driver with the
//    SESSION's durable journal home (`DurableSink`) as its authoritative sink — so the disk
//    carries the real production record stream (run header, delivered events including
//    capability-op completions, tag-12 signed frames, publishes, clocks, timer arms): a sealed
//    prefix plus an unsealed local tail, exactly the state a crash leaves. No synthetic
//    journal: the c15-20260806g corruption proved a hand-written frames-only fixture models an
//    evidence path the production ceremony does not use.
// 2. The sealed prefix publishes through the PRODUCT archive publisher
//    (`daemon_vhc_session::archive::spawn_archive_publisher`): content-addressed segments on a
//    `ContentStore` + `ArchiveHeadRecord`s attested under a base-issued run-key certificate.
// 3. The primary crashes mid-round-stream (nothing seals; no terminal is written).
// 4. `daemon_vhc_session::reconstruct::reconstruct_coordinator` — the executor the worker's
//    join path runs on a `CoordinatorRecovery` directive — re-verifies the heads against the
//    trusted base, recovers the FULL record stream (attested segments + the chained local
//    tail), re-drives it through the §8.7 input-replay engine, gates the replayed decisions
//    against the recorded publishes, and exports the rebuilt state via the typed
//    quiesce→snapshot path.
// 5. A standby boots FROM that capture (`da_migrate`) and resumes; its decisions must be
//    byte-identical to an uninterrupted reference run's — resumption, not approximation.
//
// Also pinned here: the completion-borne availability-evidence lineage (a §6.4 I6 coordinator
// whose round closure rides its OWN `payload_get` completions — the c15-20260806g shape — must
// reconstruct to the recorded round, never a silent round-0 rebirth), the CONFLICTING-HEAD
// refusal (fork evidence never reconstructs) and the untrusted-attestor refusal (a head signed
// outside the genesis trust never reconstructs).
//
// Dev/test harness: shells `cargo build` for guests; journal writes real files.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ciborium::value::Value;

use daemon_vhc_host::run::{JournalSink, RunEnd, RunIdentity};
use daemon_vhc_net::{ArchiveHeadStore, ContentStore, FsArchiveHeadStore, MemoryContentStore};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, CapabilitySet, CertScope, Hash, IrohId, PeerId,
    SigningKey, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::messages::{
    Commitment, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass,
};
use daemon_vhc_sdk_consensus::{SignedMessage, VhcMessage};
use daemon_vhc_session::archive::{spawn_archive_publisher, ArchiveSpec, SignerBinding};
use daemon_vhc_session::identity::issue_run_key;
use daemon_vhc_session::journal_home::{journal_dir, DurableSink};
use daemon_vhc_session::reconstruct::{reconstruct_coordinator, ReconstructError, ReconstructSpec};
use daemon_vhc_testkit::genesis_run::{phase_a_grants, EnvelopeInputs};
use daemon_vhc_testkit::live_genesis::fixture_authored_execution;
use daemon_vhc_testkit::{
    configure_coordinator, coordinator_state_from_capture, genesis_envelope, Coordinator,
    CoordinatorSpec,
};

const ROUNDS_BEFORE_KILL: u64 = 4;
const ROUNDS_AFTER: u64 = 2;
const RUN_LABEL: &str = "reconstruct-product";

fn coordinator_quorum_wasm() -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm("coordinator_quorum")
}

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    base.push(format!(
        "dvhc-reconstruct-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

// -- the deterministic input script (the failover drill's, verbatim) ------------------------------

struct ScriptMsg {
    key: SigningKey,
    msg: VhcMessage,
}

fn build_script(worker_keys: &[SigningKey; 2], rounds: std::ops::Range<u64>) -> Vec<ScriptMsg> {
    let peers: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();
    let mut script = Vec::new();
    if rounds.start == 0 {
        for k in worker_keys {
            script.push(ScriptMsg {
                key: k.clone(),
                msg: VhcMessage::Join(Join {
                    run_id: RUN_LABEL.into(),
                    iroh_id: IrohId([0x44; 32]),
                    class: ThroughputClass::C1,
                    capabilities: CapabilitySet::new(),
                    envelope_hash: None,
                }),
            });
        }
        for k in worker_keys {
            script.push(ScriptMsg {
                key: k.clone(),
                msg: VhcMessage::Heartbeat(Heartbeat {
                    round: 0,
                    ready: Some(true),
                }),
            });
        }
    }
    for round in rounds {
        let mut entries = Vec::new();
        for (i, k) in worker_keys.iter().enumerate() {
            let bytes = format!("update/{i}/{round}").into_bytes();
            let hash = blake3_hash(&bytes);
            script.push(ScriptMsg {
                key: k.clone(),
                msg: VhcMessage::Commitment(Commitment {
                    round,
                    payload: hash,
                    size: bytes.len() as u64,
                    locators: Vec::new(),
                }),
            });
            entries.push(RecordEntry {
                peer: peers[i],
                hash,
                size: bytes.len() as u64,
            });
        }
        script.push(ScriptMsg {
            key: worker_keys[0].clone(),
            msg: VhcMessage::StorageReceipt(StorageReceipt {
                round,
                verified: entries,
            }),
        });
    }
    script
}

fn decision_count(committed: u64, with_trailing_open: bool) -> usize {
    2 * committed as usize + usize::from(with_trailing_open)
}

fn wasm_reference(
    wasm: &[u8],
    spec: &CoordinatorSpec,
    key_seed: [u8; 32],
    script: &[ScriptMsg],
    decisions: usize,
) -> Vec<VhcMessage> {
    let mut coord = Coordinator::start(wasm, spec, phase_a_grants(), 0, key_seed).unwrap();
    for sm in script {
        coord.deliver(&sm.key, &sm.msg).expect("reference deliver");
    }
    let mut published = Vec::new();
    while published.len() < decisions {
        let (_, _, msg) = coord
            .next_decision(Duration::from_secs(60))
            .expect("reference decision");
        published.push(msg);
    }
    coord.stop().expect("reference stops clean");
    published
}

/// Encode one authoritative input in the LIVE §12.1 wire shape (`[envelope, payload, sig]`) —
/// the tag-12 evidence form the production relay journals; the reconstruction executor extracts
/// the payload structurally from it. The envelope carries the sender (the rest of the live
/// envelope is not consulted by the replay path); the payload is the canonical `VhcMessage` —
/// exactly what the live pump delivers to the module.
fn wire_frame(signed: &SignedMessage) -> (Vec<u8>, Vec<u8>) {
    let payload = to_canonical_vec(&signed.payload).expect("payload encode");
    let envelope = Value::Map(vec![(
        Value::from("sender"),
        Value::Bytes(signed.signer.0.to_vec()),
    )]);
    let frame = Value::Array(vec![
        envelope,
        Value::Bytes(payload.clone()),
        Value::Bytes(signed.sig.0.to_vec()),
    ]);
    (payload, to_canonical_vec(&frame).expect("frame encode"))
}

/// Journal one script message into the durable sink as the live relay would: the tag-12 signed
/// frame (per-sender dense seq on channel 0).
fn journal_frame(
    sink: &mut DurableSink,
    seqs: &mut std::collections::BTreeMap<[u8; 32], u64>,
    sm: &ScriptMsg,
) {
    let signed =
        SignedMessage::sign(&sm.key, VHC_PROTO_VERSION, sm.msg.clone()).expect("script sign");
    let (_payload, frame) = wire_frame(&signed);
    let sender = signed.signer.0;
    let seq = seqs.entry(sender).or_insert(0);
    sink.signed_frame(0, *seq, sender, &frame)
        .expect("journal tag-12");
    *seq += 1;
}

struct Rig {
    wasm: Vec<u8>,
    spec: CoordinatorSpec,
    base_key: SigningKey,
    base_id: PeerId,
    worker_keys: [SigningKey; 2],
}

fn rig() -> Rig {
    let wasm = coordinator_quorum_wasm();
    let coord_hash = Hash(*blake3::hash(&wasm).as_bytes());
    let base_key = SigningKey::from_bytes(blake3::hash(b"reconstruct/base-identity").as_bytes());
    let base_id = peer_id(&base_key);
    let worker_keys = [
        SigningKey::from_bytes(blake3::hash(b"reconstruct/worker/0").as_bytes()),
        SigningKey::from_bytes(blake3::hash(b"reconstruct/worker/1").as_bytes()),
    ];
    let genesis = genesis_envelope(&EnvelopeInputs {
        run_label: RUN_LABEL,
        coordinator_wasm_blake3: coord_hash,
        worker_wasm_blake3: Hash([0x77; 32]),
        coordinator_identity: base_id,
        workers: 2,
        steps_per_round: 2,
        global_batch: 4,
        execution: &fixture_authored_execution(),
    });
    let author = SigningKey::from_bytes(blake3::hash(b"reconstruct/author").as_bytes());
    let frozen = genesis.freeze(&author).expect("genesis freeze");
    let spec = configure_coordinator(&frozen).expect("coordinator configurable");
    Rig {
        wasm,
        spec,
        base_key,
        base_id,
        worker_keys,
    }
}

/// The primary's disk state after a crash: a durable journal home whose sealed prefix covers
/// `pre_cut` frames and whose unsealed local tail carries the rest. Returns the chain instance.
fn write_crashed_journal(
    root: &std::path::Path,
    run_id: Hash,
    module: Hash,
    script: &[ScriptMsg],
    pre_cut: usize,
) -> u64 {
    let identity = RunIdentity {
        run_id: run_id.0,
        epoch: 0,
        role: "coordinator".into(),
        instance: 1,
        module: module.0,
    };
    let jdir = journal_dir(root, RUN_LABEL, "coordinator", 1);
    let mut seqs = std::collections::BTreeMap::new();
    let chain_instance;
    {
        // The prefix span: journaled, then TERMINATED — the terminal seals segment 0 (the clean
        // roll the incremental publisher archives).
        let mut sink = DurableSink::open(&jdir, &identity, [0x5C; 32]).expect("journal open");
        chain_instance = sink.founding_instance();
        for sm in &script[..pre_cut] {
            journal_frame(&mut sink, &mut seqs, sm);
        }
        sink.terminal(0, Some(0), None).expect("seal the prefix");
    }
    {
        // The tail span: journaled and DROPPED unsealed — the crash cut.
        let mut sink = DurableSink::open(&jdir, &identity, [0x5C; 32]).expect("journal reopen");
        for sm in &script[pre_cut..] {
            journal_frame(&mut sink, &mut seqs, sm);
        }
        // No terminal: the tail stays unsealed on disk, exactly what a hard kill leaves.
    }
    chain_instance
}

/// Publish the sealed prefix of role incarnation `instance` through the PRODUCT publisher and
/// return the attested heads.
#[allow(clippy::too_many_arguments)]
async fn publish_prefix_of(
    root: &std::path::Path,
    run_id: Hash,
    module: Hash,
    base_key: &SigningKey,
    instance: u64,
    chain_instance: u64,
    segments: Arc<MemoryContentStore>,
    heads_dir: &std::path::Path,
) -> Vec<daemon_vhc_proto::ArchiveHeadRecord> {
    let certified = issue_run_key(
        base_key,
        CertScope {
            run_id,
            epoch: 0,
            role: "coordinator".into(),
            instance,
            module_hash: module,
        },
    )
    .expect("issue run key");
    let bindings = Arc::new(Mutex::new(vec![SignerBinding {
        signing_seed: certified.key.to_bytes(),
        certificate: certified.cert,
    }]));
    let heads: Arc<dyn ArchiveHeadStore> =
        Arc::new(FsArchiveHeadStore::open(heads_dir).await.expect("heads"));
    // Reconciliation-only: the seal stream is already closed (the primary is dead) — the
    // publisher's startup sweep publishes every sealed-but-unpublished segment.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    drop(tx);
    spawn_archive_publisher(
        RUN_LABEL.into(),
        run_id,
        "coordinator".into(),
        ArchiveSpec {
            seals: rx,
            journal_dir: journal_dir(root, RUN_LABEL, "coordinator", instance),
            chain_instance,
            round_claim: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        },
        heads.clone(),
        segments,
        bindings,
    )
    .await
    .expect("publisher drains");
    heads.fetch_heads().await.expect("stored heads")
}

/// [`publish_prefix_of`] for the founding incarnation (instance 1).
async fn publish_prefix(
    root: &std::path::Path,
    run_id: Hash,
    module: Hash,
    base_key: &SigningKey,
    chain_instance: u64,
    segments: Arc<MemoryContentStore>,
    heads_dir: &std::path::Path,
) -> Vec<daemon_vhc_proto::ArchiveHeadRecord> {
    publish_prefix_of(
        root,
        run_id,
        module,
        base_key,
        1,
        chain_instance,
        segments,
        heads_dir,
    )
    .await
}

/// Open role incarnation `instance`'s durable journal home under a SMALL rotate policy, so a
/// short drill still leaves the §8 crash shape on disk: sealed segments (the publishable
/// prefix) plus an unsealed local tail. Returns the sink and its founding chain instance.
fn open_live_sink_of(
    root: &std::path::Path,
    spec: &CoordinatorSpec,
    instance: u64,
    max_records: u64,
) -> (DurableSink, u64) {
    let identity = RunIdentity {
        run_id: spec.run_id.0,
        epoch: 0,
        role: "coordinator".into(),
        instance,
        module: spec.module_hash.0,
    };
    let jdir = journal_dir(root, RUN_LABEL, "coordinator", instance);
    let sink = DurableSink::open_with_policy(
        &jdir,
        &identity,
        [0x5C; 32],
        daemon_vhc_journal::RotatePolicy {
            max_records,
            max_open: None,
        },
    )
    .expect("journal open");
    let instance = sink.founding_instance();
    (sink, instance)
}

/// [`open_live_sink_of`] for the founding incarnation (instance 1).
fn open_live_sink(root: &std::path::Path, spec: &CoordinatorSpec) -> (DurableSink, u64) {
    open_live_sink_of(root, spec, 1, 24)
}

/// Pin the §8 crash shape the live primary left on disk: at least one sealed (publishable)
/// segment, an unsealed final tail, and every delivered tag-12 frame present inline across the
/// series — the exact input surface the reconstruction recovers.
fn assert_crash_shape(root: &std::path::Path, expected_frames: usize) {
    let jdir = journal_dir(root, RUN_LABEL, "coordinator", 1);
    let paths = daemon_vhc_journal::JournalPaths::open(&jdir).expect("journal home");
    let ordinals = paths.existing_segments().expect("segment listing");
    assert!(ordinals.len() >= 2, "the rotate policy sealed a prefix");
    let mut sealed = 0;
    let mut frames = 0;
    let mut tail_sealed = true;
    for &ord in &ordinals {
        let scan = daemon_vhc_journal::scan_file(paths.segment(ord)).expect("scan");
        sealed += usize::from(scan.sealed);
        tail_sealed = scan.sealed;
        frames += scan
            .records
            .iter()
            .filter(|r| {
                matches!(&r.body,
                    daemon_vhc_journal::Body::SignedFrame(sf) if sf.frame.is_some())
            })
            .count();
    }
    assert!(sealed >= 1, "at least one segment sealed (publishable)");
    assert!(!tail_sealed, "the final segment is the unsealed local tail");
    assert_eq!(frames, expected_frames, "every delivered frame is on disk");
}

#[tokio::test(flavor = "multi_thread")]
async fn crash_restart_reconstructs_through_the_product_path_and_resumes_byte_identically() {
    let rig = rig();
    let root = tempdir();
    let pre_kill = build_script(&rig.worker_keys, 0..ROUNDS_BEFORE_KILL);

    // -- 1. the primary drives rounds under the LIVE driver, journaling into the durable home ----
    let primary_key_seed = *blake3::hash(b"reconstruct/primary-key").as_bytes();
    let (sink, chain_instance) = open_live_sink(&root, &rig.spec);
    let mut primary = Coordinator::start_with_sink(
        &rig.wasm,
        &rig.spec,
        phase_a_grants(),
        1,
        primary_key_seed,
        Box::new(sink),
    )
    .unwrap();
    for sm in &pre_kill {
        primary.deliver(&sm.key, &sm.msg).expect("primary deliver");
    }
    let mut primary_decisions = Vec::new();
    let expected_pre = decision_count(ROUNDS_BEFORE_KILL, true);
    while primary_decisions.len() < expected_pre {
        let (_, _, msg) = primary
            .next_decision(Duration::from_secs(60))
            .expect("primary decision");
        primary_decisions.push(msg);
    }
    // The disk now carries the §8 crash shape (no terminal was ever written) — pin it.
    assert_crash_shape(&root, pre_kill.len());

    // -- 2. the sealed prefix publishes through the product publisher ----------------------------
    let segments = Arc::new(MemoryContentStore::new());
    let heads = publish_prefix(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        chain_instance,
        segments.clone(),
        &root.join("heads"),
    )
    .await;
    assert!(!heads.is_empty(), "the sealed prefix published");

    // -- 3. the crash: the primary halts with no drain, no seal, no terminal ----------------------
    // (Reconstruction runs against the exact bytes the run left behind; the guest thread is
    // reaped only after — its Stop/terminal records land past the recovery read and are inert.)

    // -- 4. the PRODUCT executor reconstructs from heads + segments + the local tail -------------
    let store: Arc<dyn ContentStore> = segments.clone();
    let capture = reconstruct_coordinator(
        ReconstructSpec {
            heads: heads.clone(),
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id],
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: Some(root.clone()),
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 1,
            restore: None,
            sidecar_key: Some([0x5C; 32]),
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect("the product reconstruction succeeds");
    let state = coordinator_state_from_capture(&capture).expect("exported state decodes");
    assert_eq!(
        state.round, ROUNDS_BEFORE_KILL,
        "the reconstructed state stands at the next un-opened round"
    );

    // -- 4b. a COLD standby (no local disk) reconstructs to the ARCHIVED point only --------------
    let store: Arc<dyn ContentStore> = segments.clone();
    let cold = reconstruct_coordinator(
        ReconstructSpec {
            heads: heads.clone(),
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id],
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: None,
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 1,
            restore: None,
            sidecar_key: None,
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect("a cold standby reconstructs from the archive alone");
    let cold_state = coordinator_state_from_capture(&cold).expect("cold state decodes");
    assert!(
        cold_state.round <= state.round,
        "the archive-only rebuild never overtakes the tail-inclusive one"
    );

    // Reap the crashed primary's guest thread (post-reconstruction: its Stop/terminal records
    // land after the recovery read and change nothing it consumed).
    let end = primary.kill().expect("primary killed");
    assert!(matches!(end, RunEnd::Outcome(_)), "guest thread joined");

    // -- 5. a standby boots FROM the reconstructed capture and resumes ---------------------------
    let standby_key_seed = *blake3::hash(b"reconstruct/standby-key").as_bytes();
    let mut standby = Coordinator::start_migrating(
        &rig.wasm,
        &rig.spec,
        phase_a_grants(),
        1,
        standby_key_seed,
        capture,
    )
    .unwrap();
    let post_kill = build_script(
        &rig.worker_keys,
        ROUNDS_BEFORE_KILL..ROUNDS_BEFORE_KILL + ROUNDS_AFTER,
    );
    for sm in &post_kill {
        standby.deliver(&sm.key, &sm.msg).expect("standby deliver");
    }
    let mut standby_decisions = Vec::new();
    let expected_post = decision_count(ROUNDS_AFTER, false);
    for i in 0..expected_post {
        let (_, _, msg) = standby
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("standby decision {i}: {e}"));
        standby_decisions.push(msg);
    }
    standby.stop().expect("standby stops clean");

    // Byte-identical to an uninterrupted reference over the full script (resumption, not
    // approximation) — the same acceptance the failover drill pins for the oracle path.
    let total_rounds = ROUNDS_BEFORE_KILL + ROUNDS_AFTER;
    let full_script = build_script(&rig.worker_keys, 0..total_rounds);
    let uninterrupted = wasm_reference(
        &rig.wasm,
        &rig.spec,
        primary_key_seed,
        &full_script,
        decision_count(total_rounds, true),
    );
    let mut resumed = primary_decisions;
    resumed.extend(standby_decisions);
    assert_eq!(uninterrupted.len(), resumed.len());
    for (a, b) in uninterrupted.iter().zip(resumed.iter()) {
        assert_eq!(
            to_canonical_vec(a).unwrap(),
            to_canonical_vec(b).unwrap(),
            "kill+reconstruct+resume ≡ uninterrupted, byte-for-byte"
        );
    }
}

/// Re-author the coordinator config with `verify_availability` ENABLED — the ceremony fleet's
/// shape (`ceremony_coordinator_config`), which the fixture-authoring default leaves off.
fn with_verify_availability(config: &[u8]) -> Vec<u8> {
    let mut v: Value = ciborium::de::from_reader(config).expect("config decodes");
    let Value::Map(entries) = &mut v else {
        panic!("coordinator config is a map");
    };
    for (k, val) in entries.iter_mut() {
        if k == &Value::from("verify_availability") {
            *val = Value::Bool(true);
        }
    }
    to_canonical_vec(&v).expect("config re-encodes")
}

/// The c15-20260806g regression (Defect 12 — the CONSENSUS-CORRUPTION shape): a coordinator
/// whose config enables availability verification (coordinator-as-storage-client, §6.4 I6 —
/// the fleet ceremony's shape) closes rounds on its OWN `payload_get` completions. No
/// `StorageReceipt` frame ever crosses the wire: the closure evidence exists ONLY as tag-14
/// completion records in the journal. The frames-only reconstruction starved of them could
/// never close a round — it silently exported a round-0 state, the join accepted it, and the
/// resumed coordinator re-ran the run from round 0 on a successor chain. The full-record
/// replay must reconstruct this lineage to the RECORDED round, and the decision gate must hold
/// (every recorded publish re-derived, in order).
#[tokio::test(flavor = "multi_thread")]
async fn completion_borne_availability_evidence_reconstructs_to_the_recorded_round() {
    const ROUNDS: u64 = 6;
    let rig = rig();
    // The ceremony config shape: availability verification ON.
    let mut spec = rig.spec.clone();
    spec.config_bytes = with_verify_availability(&rig.spec.config_bytes);
    let root = tempdir();

    // The wire carries NO StorageReceipt frames — only joins, readiness and per-round
    // commitments, sent INTO THE OPEN ROUND as live trainers do (the guest's availability
    // receipt is round-scoped at fetch completion). The committed bytes live in a content
    // table the harness serves.
    let mut payloads: std::collections::BTreeMap<[u8; 32], Vec<u8>> =
        std::collections::BTreeMap::new();
    let primary_key_seed = *blake3::hash(b"reconstruct/availability-key").as_bytes();
    let (sink, chain_instance) = open_live_sink(&root, &spec);
    let mut primary = Coordinator::start_with_sink(
        &rig.wasm,
        &spec,
        phase_a_grants(),
        1,
        primary_key_seed,
        Box::new(sink),
    )
    .unwrap();
    let mut frames = 0;
    for sm in &build_script(&rig.worker_keys, 0..0) {
        primary.deliver(&sm.key, &sm.msg).expect("primary deliver");
        frames += 1;
    }
    // Serve the guest's payload_get ops while waiting for its decisions — the §6.4 I6
    // evidence loop, journaled as completions by the real driver.
    let mut decisions = 0;
    let wait_for = |primary: &mut Coordinator,
                    payloads: &std::collections::BTreeMap<[u8; 32], Vec<u8>>,
                    decisions: &mut usize,
                    upto: usize| {
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        while *decisions < upto {
            assert!(
                std::time::Instant::now() < deadline,
                "the primary closed {decisions}/{upto} decisions before the drive deadline"
            );
            primary
                .service_payload_gets(payloads)
                .expect("storage seat services the guest's gets");
            if primary.next_decision(Duration::from_millis(100)).is_ok() {
                *decisions += 1;
            }
        }
    };
    wait_for(&mut primary, &payloads, &mut decisions, 1); // RoundOpen(0)
    for round in 0..ROUNDS {
        for (i, k) in rig.worker_keys.iter().enumerate() {
            let bytes = format!("update/{i}/{round}").into_bytes();
            let hash = blake3_hash(&bytes);
            payloads.insert(hash.0, bytes.clone());
            primary
                .deliver(
                    k,
                    &VhcMessage::Commitment(Commitment {
                        round,
                        payload: hash,
                        size: bytes.len() as u64,
                        locators: Vec::new(),
                    }),
                )
                .expect("primary deliver");
            frames += 1;
        }
        // RoundRecord(round) + RoundOpen(round + 1) close on the completion-borne receipts.
        wait_for(
            &mut primary,
            &payloads,
            &mut decisions,
            decision_count(round + 1, true),
        );
    }
    assert_crash_shape(&root, frames);

    // The journal IS the c15-20260806g shape: closure evidence exists ONLY as completions.
    {
        let jdir = journal_dir(&root, RUN_LABEL, "coordinator", 1);
        let paths = daemon_vhc_journal::JournalPaths::open(&jdir).expect("journal home");
        let mut completions = 0;
        for ord in paths.existing_segments().expect("segment listing") {
            let scan = daemon_vhc_journal::scan_file(paths.segment(ord)).expect("scan");
            for r in &scan.records {
                match &r.body {
                    daemon_vhc_journal::Body::Completion(_) => completions += 1,
                    daemon_vhc_journal::Body::SignedFrame(sf) => {
                        // The harness relay journals the evidence as a canonical SignedMessage.
                        let frame = sf.frame.as_deref().expect("tag-12 frame inline");
                        let signed: SignedMessage = daemon_vhc_proto::from_canonical_slice(frame)
                            .expect("journaled evidence decodes");
                        assert!(
                            !matches!(signed.payload, VhcMessage::StorageReceipt(_)),
                            "no receipt frame may exist in this lineage — the evidence is \
                             completion-borne by construction"
                        );
                    }
                    _ => {}
                }
            }
        }
        assert!(
            completions >= (ROUNDS * 2) as usize,
            "one journaled payload_get completion per commitment, got {completions}"
        );
    }

    let segments = Arc::new(MemoryContentStore::new());
    let heads = publish_prefix(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        chain_instance,
        segments.clone(),
        &root.join("heads"),
    )
    .await;
    assert!(!heads.is_empty(), "the sealed prefix published");

    let store: Arc<dyn ContentStore> = segments;
    let capture = reconstruct_coordinator(
        ReconstructSpec {
            heads,
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id],
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: Some(root),
            module: rig.wasm.clone(),
            config: spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 1,
            restore: None,
            sidecar_key: Some([0x5C; 32]),
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect("a completion-borne availability lineage reconstructs");
    let state = coordinator_state_from_capture(&capture).expect("exported state decodes");
    assert_eq!(
        state.round, ROUNDS,
        "the reconstructed state stands at the RECORDED next un-opened round — \
         never a silent round-0 rebirth"
    );

    let end = primary.kill().expect("primary killed");
    assert!(matches!(end, RunEnd::Outcome(_)), "guest thread joined");
}

/// Fork evidence never reconstructs: two attested heads at the same height that do not extend
/// one another refuse typed at the worker's re-verification (§8.8 [AR-4]) — reconstruction is
/// the LAST place a fork may be papered over.
#[tokio::test(flavor = "multi_thread")]
async fn conflicting_heads_refuse_reconstruction_typed() {
    let rig = rig();
    let root = tempdir();
    let pre_kill = build_script(&rig.worker_keys, 0..ROUNDS_BEFORE_KILL);
    let chain_instance = write_crashed_journal(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &pre_kill,
        pre_kill.len() * 2 / 3,
    );
    let segments = Arc::new(MemoryContentStore::new());
    let mut heads = publish_prefix(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        chain_instance,
        segments.clone(),
        &root.join("heads"),
    )
    .await;

    // A conflicting head at height 0 of the SAME chain scope: identical coordinates, different
    // segment hash — signed by the same trusted base (the strongest fork shape: authorization
    // alone cannot reject it; only the chain fold can).
    let certified = issue_run_key(
        &rig.base_key,
        CertScope {
            run_id: rig.spec.run_id,
            epoch: 0,
            role: "coordinator".into(),
            instance: 1,
            module_hash: rig.spec.module_hash,
        },
    )
    .expect("issue run key");
    let mut forked_body = heads[0].body.clone();
    forked_body.segment_hash = Hash([0xF0; 32]);
    let forked =
        daemon_vhc_proto::ArchiveHeadRecord::publish(&certified.key, certified.cert, forked_body)
            .expect("fork head authors");
    heads.push(forked);

    let store: Arc<dyn ContentStore> = segments;
    let err = reconstruct_coordinator(
        ReconstructSpec {
            heads,
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id],
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: Some(root),
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 1,
            restore: None,
            sidecar_key: None,
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect_err("fork evidence must refuse reconstruction");
    assert!(
        matches!(err, ReconstructError::Verify(_)),
        "the refusal is the typed chain-verification error, got: {err}"
    );
}

/// A head attested OUTSIDE the genesis trust never reconstructs: the worker re-verifies the
/// carried heads itself (carriage is bootstrap, not trust) — a node compromise cannot smuggle
/// an unattested history into the seat.
#[tokio::test(flavor = "multi_thread")]
async fn untrusted_attestor_refuses_reconstruction_typed() {
    let rig = rig();
    let root = tempdir();
    let pre_kill = build_script(&rig.worker_keys, 0..ROUNDS_BEFORE_KILL);
    let chain_instance = write_crashed_journal(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &pre_kill,
        pre_kill.len() * 2 / 3,
    );
    // Published under a base the GENESIS does not trust.
    let rogue = SigningKey::from_bytes(blake3::hash(b"reconstruct/rogue-base").as_bytes());
    let segments = Arc::new(MemoryContentStore::new());
    let heads = publish_prefix(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rogue,
        chain_instance,
        segments.clone(),
        &root.join("heads"),
    )
    .await;
    assert!(!heads.is_empty());

    let store: Arc<dyn ContentStore> = segments;
    let err = reconstruct_coordinator(
        ReconstructSpec {
            heads,
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id], // the genesis trust — not the rogue
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: Some(root),
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 1,
            restore: None,
            sidecar_key: None,
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect_err("an untrusted attestation must refuse reconstruction");
    assert!(
        matches!(err, ReconstructError::Verify(_)),
        "the refusal is the typed chain-verification error, got: {err}"
    );
}

/// The c15-20260806h terminal shape (Defect 13): after a first crash + reconstruction the
/// standby resumes via §10.2 migration — and the successor chain's first records are its
/// restore read-backs, whose values exceed `READBACK_INLINE_MAX` and ride ENCRYPTED LOCAL
/// SIDECARS (§8.5; the key material never rides the archive). c15h's SECOND reconstruction
/// refused on exactly those references ("no sidecar key material rides the archive"), five
/// retries consumed the budget, and the run went terminal on a healthy box. Same-box, the
/// node's own journal key must decrypt the recorded sections (with the content-addressed
/// migration-capture fallback for what does not hydrate) and the two-chain lineage must
/// reconstruct to the recorded round.
#[tokio::test(flavor = "multi_thread")]
async fn a_resumed_seat_with_sidecar_restore_readbacks_reconstructs_after_a_second_crash() {
    // Enough committed rounds that the exported CoordinatorState exceeds READBACK_INLINE_MAX
    // (the retained record ring grows per round) — asserted below, not assumed.
    const PRE: u64 = 10;
    const POST: u64 = 2;
    let rig = rig();
    let root = tempdir();

    // -- crash #1: the primary drives PRE rounds under the live driver, then dies ----------------
    let primary_key_seed = *blake3::hash(b"reconstruct/second-crash/primary").as_bytes();
    let (sink, chain1) = open_live_sink(&root, &rig.spec);
    let mut primary = Coordinator::start_with_sink(
        &rig.wasm,
        &rig.spec,
        phase_a_grants(),
        1,
        primary_key_seed,
        Box::new(sink),
    )
    .unwrap();
    for sm in &build_script(&rig.worker_keys, 0..PRE) {
        primary.deliver(&sm.key, &sm.msg).expect("primary deliver");
    }
    for i in 0..decision_count(PRE, true) {
        primary
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("primary decision {i}: {e}"));
    }

    // ONE head store for the whole lineage (the production registry): the successor chain's
    // publisher resolves its succession link from the store's view of earlier chains.
    let heads_dir = root.join("heads");
    let segments = Arc::new(MemoryContentStore::new());
    let heads = publish_prefix(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        chain1,
        segments.clone(),
        &heads_dir,
    )
    .await;
    assert!(!heads.is_empty(), "chain 1's sealed prefix published");

    let store: Arc<dyn ContentStore> = segments.clone();
    let capture = reconstruct_coordinator(
        ReconstructSpec {
            heads: heads.clone(),
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id],
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: Some(root.clone()),
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 2,
            restore: None,
            sidecar_key: Some([0x5C; 32]),
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect("the first-crash reconstruction succeeds");
    let end = primary.kill().expect("primary killed");
    assert!(matches!(end, RunEnd::Outcome(_)), "guest thread joined");

    // -- the standby resumes FROM the capture (§10.2), journaling to its own durable home --------
    let standby_key_seed = *blake3::hash(b"reconstruct/second-crash/standby").as_bytes();
    // A tight rotate policy: the successor's FIRST segment (the one carrying the §10.2 restore
    // read-back) must seal and publish before the short post-resume drive ends.
    let (sink2, chain2) = open_live_sink_of(&root, &rig.spec, 2, 8);
    let mut standby = Coordinator::start_migrating_with_sink(
        &rig.wasm,
        &rig.spec,
        phase_a_grants(),
        2,
        standby_key_seed,
        capture,
        Box::new(sink2),
    )
    .unwrap();
    for sm in &build_script(&rig.worker_keys, PRE..PRE + POST) {
        standby.deliver(&sm.key, &sm.msg).expect("standby deliver");
    }
    for i in 0..decision_count(POST, false) {
        standby
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("standby decision {i}: {e}"));
    }

    // The defect shape MUST be on disk: the successor chain's §10.2 restore read-back rode a
    // sidecar (value NOT inline in the record) — otherwise this drill proves nothing.
    {
        let jdir = journal_dir(&root, RUN_LABEL, "coordinator", 2);
        let paths = daemon_vhc_journal::JournalPaths::open(&jdir).expect("journal home");
        let mut sidecar_refs = 0;
        for ord in paths.existing_segments().expect("segment listing") {
            let scan = daemon_vhc_journal::scan_file(paths.segment(ord)).expect("scan");
            for r in &scan.records {
                if let daemon_vhc_journal::Body::ReadBack(rb) = &r.body {
                    if rb.sidecar.is_some() && rb.value.is_none() {
                        sidecar_refs += 1;
                    }
                }
            }
        }
        assert!(
            sidecar_refs >= 1,
            "the restore read-back must ride a sidecar (§8.5) for this drill to exercise \
             the second-crash shape — raise PRE if the state stayed under the inline max"
        );
    }

    // -- crash #2: chain 2's sealed prefix publishes; reconstruct over BOTH chains ---------------
    let heads = publish_prefix_of(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        2,
        chain2,
        segments.clone(),
        &heads_dir,
    )
    .await;
    assert!(
        heads.iter().any(|h| h.body.chain_instance == chain2),
        "the successor chain's sealed prefix published into the shared store"
    );
    let store: Arc<dyn ContentStore> = segments;
    let capture2 = reconstruct_coordinator(
        ReconstructSpec {
            heads,
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id],
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: Some(root.clone()),
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 3,
            restore: None,
            sidecar_key: Some([0x5C; 32]),
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect("the second-crash lineage reconstructs (Defect 13: sidecar restore read-backs)");
    let state = coordinator_state_from_capture(&capture2).expect("exported state decodes");
    assert_eq!(
        state.round,
        PRE + POST,
        "the rebuilt seat stands at the RECORDED next un-opened round across both chains"
    );

    let end = standby.kill().expect("standby killed");
    assert!(
        matches!(end, RunEnd::Outcome(_)),
        "standby guest thread joined"
    );
}

/// Seal a crashed chain's unsealed tail by the SAME-BOX recovery reopen (the production
/// mechanism that closes the archive/crash gap: the next incarnation reopens the file series,
/// the recovery point seals, and the publisher's startup sweep archives it). The terminal is
/// a journal record, never a replay input.
fn seal_tail(root: &std::path::Path, spec: &CoordinatorSpec, instance: u64) {
    let identity = RunIdentity {
        run_id: spec.run_id.0,
        epoch: 0,
        role: "coordinator".into(),
        instance,
        module: spec.module_hash.0,
    };
    let jdir = journal_dir(root, RUN_LABEL, "coordinator", instance);
    let mut sink = DurableSink::open(&jdir, &identity, [0x5C; 32]).expect("recovery reopen");
    sink.terminal(2, None, None)
        .expect("seal the recovery point");
}

/// **The Gate A decisive regression (defect 13 closed properly): archive-portable recovery
/// with ZERO local state.** The plan's bar: "delete ALL local run state (journals, sidecars,
/// `journal.key`), reconstruct solely from the assembled product archive, reproduce the
/// committed head and every recorded decision."
///
/// The lineage crosses the hardest seam — a successor chain whose first records are §10.2
/// restore read-backs riding encrypted LOCAL sidecars (the c15h refusal shape) — and the
/// final reconstruction runs as a true cold standby: `journal_root: None` (no local segments,
/// no local tail, no sidecar files) and `sidecar_key: None` (no node key material). The
/// kind-3 values must resolve through the AUTHORITATIVE path: content-addressed against the
/// migration capture rebuilt from the archived record stream itself. The decision gate
/// (`verify_decisions`) runs inside the executor at every span, so success here IS the
/// "every recorded decision" proof.
#[tokio::test(flavor = "multi_thread")]
async fn a_cold_standby_with_zero_local_state_reconstructs_solely_from_the_product_archive() {
    const PRE: u64 = 10;
    const POST: u64 = 2;
    let rig = rig();
    let root = tempdir();

    // -- chain 1: the primary drives PRE rounds, crashes; the recovery reopen seals the tail ------
    let primary_key_seed = *blake3::hash(b"reconstruct/zero-local/primary").as_bytes();
    let (sink, chain1) = open_live_sink(&root, &rig.spec);
    let mut primary = Coordinator::start_with_sink(
        &rig.wasm,
        &rig.spec,
        phase_a_grants(),
        1,
        primary_key_seed,
        Box::new(sink),
    )
    .unwrap();
    for sm in &build_script(&rig.worker_keys, 0..PRE) {
        primary.deliver(&sm.key, &sm.msg).expect("primary deliver");
    }
    for i in 0..decision_count(PRE, true) {
        primary
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("primary decision {i}: {e}"));
    }
    let end = primary.kill().expect("primary killed");
    assert!(matches!(end, RunEnd::Outcome(_)), "guest thread joined");
    seal_tail(&root, &rig.spec, 1);

    let heads_dir = root.join("heads");
    let segments = Arc::new(MemoryContentStore::new());
    let heads = publish_prefix(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        chain1,
        segments.clone(),
        &heads_dir,
    )
    .await;
    assert!(!heads.is_empty(), "chain 1 published to the archive");

    // -- reconstruction #1 is ALREADY a cold standby: the successor may be a different box, so
    //    the capture it resumes from must derive from the archive alone --------------------------
    let store: Arc<dyn ContentStore> = segments.clone();
    let capture = reconstruct_coordinator(
        ReconstructSpec {
            heads: heads.clone(),
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id],
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: None,
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 2,
            restore: None,
            sidecar_key: None,
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect("the first cold-standby reconstruction succeeds from the archive alone");

    // -- chain 2: the standby resumes from the capture (§10.2), drives POST rounds, crashes ------
    let standby_key_seed = *blake3::hash(b"reconstruct/zero-local/standby").as_bytes();
    let (sink2, chain2) = open_live_sink_of(&root, &rig.spec, 2, 8);
    let mut standby = Coordinator::start_migrating_with_sink(
        &rig.wasm,
        &rig.spec,
        phase_a_grants(),
        2,
        standby_key_seed,
        capture,
        Box::new(sink2),
    )
    .unwrap();
    for sm in &build_script(&rig.worker_keys, PRE..PRE + POST) {
        standby.deliver(&sm.key, &sm.msg).expect("standby deliver");
    }
    for i in 0..decision_count(POST, false) {
        standby
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("standby decision {i}: {e}"));
    }
    let end = standby.kill().expect("standby killed");
    assert!(
        matches!(end, RunEnd::Outcome(_)),
        "standby guest thread joined"
    );
    seal_tail(&root, &rig.spec, 2);

    // The defect shape MUST be on disk before the sweep: the successor chain's §10.2 restore
    // read-back rode a sidecar — otherwise this drill proves nothing about Gate A.
    {
        let jdir = journal_dir(&root, RUN_LABEL, "coordinator", 2);
        let paths = daemon_vhc_journal::JournalPaths::open(&jdir).expect("journal home");
        let mut sidecar_refs = 0;
        for ord in paths.existing_segments().expect("segment listing") {
            let scan = daemon_vhc_journal::scan_file(paths.segment(ord)).expect("scan");
            for r in &scan.records {
                if let daemon_vhc_journal::Body::ReadBack(rb) = &r.body {
                    if rb.sidecar.is_some() && rb.value.is_none() {
                        sidecar_refs += 1;
                    }
                }
            }
        }
        assert!(
            sidecar_refs >= 1,
            "the restore read-back must ride a sidecar (§8.5) for this drill to exercise \
             the archive-portability seam — raise PRE if the state stayed under the inline max"
        );
    }

    let heads = publish_prefix_of(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        2,
        chain2,
        segments.clone(),
        &heads_dir,
    )
    .await;
    assert!(
        heads.iter().any(|h| h.body.chain_instance == chain2),
        "chain 2 published to the archive"
    );

    // -- DELETE all local run state: journals, sidecars, unsealed tails, everything --------------
    let run_state = daemon_vhc_session::journal_home::run_state_dir(&root, RUN_LABEL);
    assert!(
        run_state.is_dir(),
        "the run-state dir exists before the sweep"
    );
    std::fs::remove_dir_all(&run_state).expect("local run state deleted");

    // -- the decisive reconstruction: zero local state, no key material, archive only ------------
    let store: Arc<dyn ContentStore> = segments;
    let capture2 = reconstruct_coordinator(
        ReconstructSpec {
            heads,
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id],
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: None,
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 3,
            restore: None,
            sidecar_key: None,
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect("zero local state: the product archive alone reconstructs the seat (Gate A)");
    let state = coordinator_state_from_capture(&capture2).expect("exported state decodes");
    assert_eq!(
        state.round,
        PRE + POST,
        "the rebuilt seat stands at the RECORDED next un-opened round — the committed head \
         reproduced solely from the archive"
    );
}
