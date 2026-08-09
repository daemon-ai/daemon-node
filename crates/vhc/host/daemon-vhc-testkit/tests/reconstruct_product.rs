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

use daemon_vhc_host::run::{
    Dropped, JournalSink, RunEnd, RunHeaderResources, RunIdentity, SinkError,
};
use daemon_vhc_net::{
    ArchiveHeadStore, ContentStore, FsArchiveHeadStore, FsContentStore, MemoryContentStore,
};
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
use daemon_vhc_session::reconstruct::{
    certify_lineage, extract_catch_up_frames, reconstruct_coordinator, CatchUpSpec, ClosureClass,
    ReconstructError, ReconstructSpec,
};
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
/// return the attested heads. `trusted` is the genesis-trusted attestor set the publisher may
/// link cross-base predecessors against (empty = own-base linking only).
#[allow(clippy::too_many_arguments)]
async fn publish_prefix_of(
    root: &std::path::Path,
    run_id: Hash,
    module: Hash,
    base_key: &SigningKey,
    instance: u64,
    chain_instance: u64,
    segments: Arc<dyn ContentStore>,
    heads_dir: &std::path::Path,
    trusted: Vec<daemon_vhc_proto::PeerId>,
    predecessors: Vec<daemon_vhc_session::archive::PredecessorChain>,
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
            archived_round: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            trusted,
            predecessors,
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
    segments: Arc<dyn ContentStore>,
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
        Vec::new(),
        Vec::new(),
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
            ..daemon_vhc_journal::RotatePolicy::default()
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
        Vec::new(),
        Vec::new(),
    )
    .await;
    assert!(
        heads.iter().any(|h| h.body.chain_instance == chain2),
        "the successor chain's sealed prefix published into the shared store"
    );
    let store: Arc<dyn ContentStore> = segments.clone();
    let capture2 = reconstruct_coordinator(
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

    // -- certification (matrix case 3): the SIDECAR-backed migration capture certifies -----------
    // The successor's §10.2 restore read-back rode a sidecar (asserted above), so the seam gate
    // binds it BY HASH against the section staged at that identity — the by-ref half of the
    // kind-3 check. The successor still runs (its tail is an unsealed prefix), so the closure
    // class is PREFIX, and completeness is not claimed.
    let report = certify_lineage(
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
            incarnation: 4,
            restore: None,
            sidecar_key: Some([0x5C; 32]),
            deadline_ms: 60_000,
        },
        segments,
    )
    .await
    .expect("a sidecar-backed migration capture certifies GREEN");
    assert_eq!(report.seams.len(), 1, "the migration seam is bound");
    assert!(
        report.seams[0].kind3_checked >= 1,
        "the sidecar-borne restore read-back was bound to the staged section"
    );
    assert_eq!(
        report.closure,
        ClosureClass::Prefix,
        "a still-running lineage is a verified prefix, never claimed complete"
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
        Vec::new(),
        Vec::new(),
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

/// **The defect-16 regression (c15k, 2026-08-08): a crash tail CONSUMED by reconstruction must
/// reach the archive.** The live shape: the killed coordinator left a 99-record suffix past its
/// last archived head; the same-box reconstruction consumed it (correct — the successor's boot
/// capture folds it), but nothing ever published it, so every later archive fold replayed a
/// state BEHIND the successor's recorded §10.2 restore read-back and refused at the
/// content-address gate ("resolves neither inline nor against the span's migration capture").
///
/// The closure under test, end to end on the production paths:
/// 1. reconstruction SEALS the abandoned tail in place before consuming it
///    (`seal_abandoned_tail` inside `recover_records`);
/// 2. the successor session's archive publisher ADOPTS the predecessor's
///    sealed-but-unpublished suffix — uploading the segments and attesting their heads under
///    its OWN certified span (cross-span attestation) — BEFORE its founding head commits the
///    succession link;
/// 3. a zero-local-state reconstruction over the completed archive rebuilds the whole lineage.
#[tokio::test(flavor = "multi_thread")]
async fn a_consumed_crash_tail_reaches_the_archive_and_the_lineage_reconstructs_archive_only() {
    const PRE: u64 = 10;
    const POST: u64 = 2;
    let rig = rig();
    let root = tempdir();

    // -- chain 1: the primary drives PRE rounds, killed — an UNSEALED tail with records ----------
    // A rotate cadence chosen so the kill cuts MID-segment: the tail must carry records the
    // sealed prefix does not (the c15k shape — 99 consumed-but-unarchived records).
    let primary_key_seed = *blake3::hash(b"reconstruct/defect16/primary").as_bytes();
    let (sink, chain1) = open_live_sink_of(&root, &rig.spec, 1, 7);
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

    // The hard kill: snapshot the journal bytes AT the cut (the graceful `kill()` below writes
    // a Stop terminal that seals — the live c15k kill was `SIGKILL`, which does not), reap the
    // guest thread, then restore the snapshot. The restored directory is byte-for-byte what a
    // hard-killed process leaves behind.
    let jdir1 = journal_dir(&root, RUN_LABEL, "coordinator", 1);
    let crash_copy = root.join("crash-cut-chain1");
    copy_dir(&jdir1, &crash_copy);
    let end = primary.kill().expect("primary killed");
    assert!(matches!(end, RunEnd::Outcome(_)), "guest thread joined");
    std::fs::remove_dir_all(&jdir1).expect("discard the post-kill journal");
    std::fs::rename(&crash_copy, &jdir1).expect("restore the crash-cut bytes");

    // Pin the c15k shape: the final segment is UNSEALED and carries records (the crash cut).
    let tail_ord = {
        let paths = daemon_vhc_journal::JournalPaths::open(&jdir1).expect("journal home");
        let ords = paths.existing_segments().expect("segment listing");
        let last = *ords.last().expect("at least one segment");
        let scan = daemon_vhc_journal::scan_file(paths.segment(last)).expect("scan");
        assert!(
            !scan.sealed && !scan.records.is_empty(),
            "the crash must leave an unsealed tail WITH records for this drill \
             (rotate-policy drift? tail: sealed={} records={})",
            scan.sealed,
            scan.records.len()
        );
        last
    };

    // Only the SEALED prefix reaches the archive pre-crash (the tail is local-only — c15k).
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
    assert!(
        heads
            .iter()
            .filter(|h| h.body.chain_instance == chain1)
            .all(|h| h.body.segment < tail_ord),
        "the unsealed tail must NOT be in the pre-crash archive"
    );

    // -- reconstruction #1 (same box): consumes prefix + tail, SEALING the tail in place ---------
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
    .expect("the first (same-box) reconstruction succeeds");
    {
        let paths = daemon_vhc_journal::JournalPaths::open(&jdir1).expect("journal home");
        let scan = daemon_vhc_journal::scan_file(paths.segment(tail_ord)).expect("scan");
        assert!(
            scan.sealed,
            "the consumed crash tail was sealed in place (Gate A durability ordering)"
        );
    }

    // -- chain 2: the standby resumes from the capture, drives POST rounds, killed ---------------
    let standby_key_seed = *blake3::hash(b"reconstruct/defect16/standby").as_bytes();
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

    // -- the successor's publisher ADOPTS chain 1's backlog, then publishes chain 2 --------------
    let heads = publish_prefix_of(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        2,
        chain2,
        segments.clone(),
        &heads_dir,
        Vec::new(),
        vec![daemon_vhc_session::archive::PredecessorChain {
            chain_instance: chain1,
            journal_dir: jdir1.clone(),
        }],
    )
    .await;
    let chain1_tip = heads
        .iter()
        .filter(|h| h.body.chain_instance == chain1)
        .max_by_key(|h| h.body.segment)
        .expect("chain 1 stored");
    assert_eq!(
        chain1_tip.body.segment, tail_ord,
        "the adopted crash-tail head reached the store"
    );
    assert_eq!(
        chain1_tip.body.instance, 2,
        "the tail head is attested by the SUCCESSOR span (cross-span attestation)"
    );
    let founding2 = heads
        .iter()
        .find(|h| h.body.chain_instance == chain2 && h.body.segment == 0)
        .expect("chain 2 founding head");
    assert_eq!(
        founding2.body.predecessor,
        Some(
            chain1_tip
                .content_address()
                .expect("terminal head re-encodes")
        ),
        "the successor's succession link names the COMPLETE predecessor terminal (the \
         adopted tail head), not the stale pre-crash tip"
    );

    // -- zero local state: the archive alone rebuilds the whole lineage --------------------------
    let run_state = daemon_vhc_session::journal_home::run_state_dir(&root, RUN_LABEL);
    std::fs::remove_dir_all(&run_state).expect("local run state deleted");
    let store: Arc<dyn ContentStore> = segments.clone();
    let capture2 = reconstruct_coordinator(
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
            incarnation: 3,
            restore: None,
            sidecar_key: None,
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect(
        "defect 16 closed: the consumed tail is IN the archive, so the lineage reconstructs \
         with zero local state",
    );
    let state = coordinator_state_from_capture(&capture2).expect("exported state decodes");
    assert_eq!(
        state.round,
        PRE + POST,
        "the rebuilt seat stands at the recorded next un-opened round across both chains"
    );

    // -- the certification kernel over the SAME lineage (matrix cases 1, 5, 6 + closure class) ---
    let certify_spec =
        |heads: Vec<daemon_vhc_proto::ArchiveHeadRecord>, incarnation: u64| ReconstructSpec {
            heads,
            run_id: rig.spec.run_id,
            trusted: vec![rig.base_id],
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: None,
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation,
            restore: None,
            sidecar_key: None,
            deadline_ms: 60_000,
        };

    // GREEN (case 1): the same-base two-chain lineage certifies, the seam bound, the closure
    // class reported. The final span carries the kind-0 terminal the graceful stop journaled,
    // so certification closes TERMINAL — the archive is a complete record, not merely a prefix.
    let report = certify_lineage(certify_spec(heads.clone(), 4), segments.clone())
        .await
        .expect("the two-chain lineage certifies GREEN");
    assert_eq!(report.spans.len(), 2, "two incarnation spans replayed");
    assert_eq!(report.seams.len(), 1, "exactly one reason-2 seam, bound");
    assert!(
        report.seams[0].kind3_checked >= 1,
        "the successor's restore read-backs were bound to the predecessor's export \
         (checked {})",
        report.seams[0].kind3_checked
    );
    assert!(
        matches!(report.closure, ClosureClass::Terminal { .. }),
        "a lineage ending at a recorded kind-0 terminal closes TERMINAL, got {:?}",
        report.closure
    );

    // RED (case 5): the consumed crash tail withheld from the head snapshot. The founding
    // successor head's succession pointer names the ADOPTED tail head by content address, so a
    // snapshot missing it cannot even splice the lineage — the typed refusal fires BEFORE any
    // replay. (Were the linkage somehow intact, the span-1 seam anchor gate is the backstop:
    // the truncated predecessor's export can never match the successor's anchoring tag-10 —
    // proven at the unit level in `daemon-vhc-session::reconstruct::seam_tests`.)
    let sans_tail: Vec<_> = heads
        .iter()
        .filter(|h| !(h.body.chain_instance == chain1 && h.body.segment == tail_ord))
        .cloned()
        .collect();
    let err = certify_lineage(certify_spec(sans_tail, 5), segments.clone())
        .await
        .expect_err("a lineage missing its consumed crash tail must refuse");
    assert!(
        matches!(err, ReconstructError::Verify(_)),
        "the succession link surfaces the missing tail as a typed lineage refusal: {err}"
    );

    // RED (case 6): every predecessor segment address broken (an empty content plane, zero
    // local state) — the typed segment refusal fires before any replay, naming the chain.
    let empty: Arc<dyn ContentStore> = Arc::new(MemoryContentStore::new());
    let err = certify_lineage(certify_spec(heads.clone(), 6), empty)
        .await
        .expect_err("unresolvable segment addresses must refuse before replay");
    assert!(
        matches!(err, ReconstructError::Segment { .. }),
        "a broken predecessor address is a typed segment refusal: {err}"
    );

    // GREEN closure regression: the same lineage truncated BEFORE the recorded terminal (every
    // chain-2 segment from the kind-0 terminal onward withheld) still certifies — an archive is
    // a sealed prefix — but the kernel reports PREFIX closure: the recorded terminal is gone,
    // so completeness is not claimed.
    let mut terminal_seg = None;
    for h in heads.iter().filter(|h| h.body.chain_instance == chain2) {
        let bytes = segments
            .get_content(&h.body.segment_hash)
            .await
            .expect("attested segment bytes");
        let scan = daemon_vhc_journal::scan_bytes(&bytes).expect("scan");
        let has_terminal = scan
            .records
            .iter()
            .any(|r| matches!(&r.body, daemon_vhc_journal::Body::Terminal(t) if t.kind == 0));
        if has_terminal {
            terminal_seg =
                Some(terminal_seg.map_or(h.body.segment, |s: u64| s.min(h.body.segment)));
        }
    }
    let terminal_seg = terminal_seg.expect("the completed chain records its kind-0 terminal");
    assert!(
        terminal_seg > 0,
        "the terminal must not sit in the founding segment for the truncation to leave a chain"
    );
    let truncated: Vec<_> = heads
        .iter()
        .filter(|h| !(h.body.chain_instance == chain2 && h.body.segment >= terminal_seg))
        .cloned()
        .collect();
    let report = certify_lineage(certify_spec(truncated, 7), segments.clone())
        .await
        .expect("a truncated archive is a verified sealed prefix — GREEN");
    assert_eq!(
        report.closure,
        ClosureClass::Prefix,
        "with the terminal segment withheld the closure class degrades to PREFIX"
    );
}

/// A flat file copy (journal homes hold no subdirectories) — the hard-kill byte snapshot.
fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).expect("snapshot dir");
    for entry in std::fs::read_dir(from).expect("read journal home") {
        let path = entry.expect("dir entry").path();
        if path.is_file() {
            std::fs::copy(&path, to.join(path.file_name().expect("file name")))
                .expect("snapshot file");
        }
    }
}

/// Hex of a proto hash (the [`FsContentStore`] object filename — objects live flat as
/// `<root>/<blake3 hex>`).
fn hex_of(h: &Hash) -> String {
    let mut s = String::with_capacity(64);
    for b in h.0 {
        s.push(char::from_digit((b >> 4) as u32, 16).expect("nibble"));
        s.push(char::from_digit((b & 0xf) as u32, 16).expect("nibble"));
    }
    s
}

/// **Gate B'/E: the trainer catch-up bridge over REAL production journals, under round-aware
/// seal pacing, ending at the fs-plane retention wall.** The c15h wedge shape (a trainer fence
/// rounds behind the live head): the coordinator drives ten rounds under the live driver with
/// seal pacing as the ONLY rotation input (no count/age threshold — publication is maximally
/// delayed, the reconciliation sweep publishes everything at the end, and the recovery points
/// exist purely because pacing requested them). A cold rejoiner with fence 4 then extracts the
/// staged catch-up frames from the attested archive alone and receives EXACTLY the post-fence
/// committed `RoundRecord`s, ascending, each carrying the guest's genuine module payload. Then
/// the retention wall: prune one archived segment from the fs content plane and the same
/// extraction refuses with the TYPED segment error (the re-scoped `CheckpointStale` posture —
/// a genuine archive gap is loud, never a silent wedge).
#[tokio::test(flavor = "multi_thread")]
async fn a_trainer_fence_past_the_ring_bridges_from_the_paced_archive_until_the_retention_wall() {
    const PRE: u64 = 10;
    const FENCE: u64 = 4;
    let rig = rig();
    let root = tempdir();

    // -- the primary drives PRE rounds; seal pacing is the only rotation input -------------------
    let pacer_cell = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let identity = RunIdentity {
        run_id: rig.spec.run_id.0,
        epoch: 0,
        role: "coordinator".into(),
        instance: 1,
        module: rig.spec.module_hash.0,
    };
    let jdir = journal_dir(&root, RUN_LABEL, "coordinator", 1);
    let sink = DurableSink::open_with_policy(
        &jdir,
        &identity,
        [0x5C; 32],
        daemon_vhc_journal::RotatePolicy {
            max_records: 100_000, // never reached — recovery points come from pacing alone
            max_open: None,
            roll_request: Some(pacer_cell.clone()),
        },
    )
    .expect("paced journal open");
    let chain1 = sink.founding_instance();
    let primary_key_seed = *blake3::hash(b"reconstruct/catch-up/primary").as_bytes();
    let mut primary = Coordinator::start_with_sink(
        &rig.wasm,
        &rig.spec,
        phase_a_grants(),
        1,
        primary_key_seed,
        Box::new(sink),
    )
    .unwrap();
    let mut decisions = 0usize;
    for round in 0..PRE {
        // Round 0's script carries the join/readiness preamble; later rounds are commitments +
        // the receipt only (`build_script` keys the preamble off `rounds.start == 0`).
        for sm in &build_script(&rig.worker_keys, round..round + 1) {
            primary.deliver(&sm.key, &sm.msg).expect("primary deliver");
        }
        while decisions < decision_count(round + 1, true) {
            primary
                .next_decision(Duration::from_secs(60))
                .unwrap_or_else(|e| panic!("primary decision {decisions}: {e}"));
            decisions += 1;
        }
        // The SealPacer's move: the committed-round watermark drifted past the archive tip —
        // request a recovery point (honored at the next append, exactly like production).
        if round % 2 == 1 {
            pacer_cell.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let end = primary.kill().expect("primary killed");
    assert!(matches!(end, RunEnd::Outcome(_)), "guest thread joined");

    // Pacing (not count, not age) produced the sealed prefix.
    {
        let paths = daemon_vhc_journal::JournalPaths::open(&jdir).expect("journal home");
        let ordinals = paths.existing_segments().expect("segment listing");
        let sealed = ordinals
            .iter()
            .filter(|&&ord| {
                daemon_vhc_journal::scan_file(paths.segment(ord))
                    .expect("scan")
                    .sealed
            })
            .count();
        assert!(
            sealed >= 3,
            "round-aware seal pacing produced the archive prefix (sealed={sealed})"
        );
    }
    seal_tail(&root, &rig.spec, 1);

    // -- the sealed chain publishes through the product publisher onto the FS content plane ------
    let seg_root = root.join("segments");
    let fs_plane = Arc::new(FsContentStore::open(&seg_root).expect("fs content plane"));
    let heads = publish_prefix(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        chain1,
        fs_plane.clone(),
        &root.join("heads"),
    )
    .await;
    assert!(heads.len() >= 4, "every paced recovery point published");

    // -- the cold rejoiner (no local disk) extracts the staged catch-up past its fence ------------
    let spec = CatchUpSpec {
        heads: heads.clone(),
        run_id: rig.spec.run_id,
        trusted: vec![rig.base_id],
        run_label: RUN_LABEL.into(),
        journal_root: None,
        after_round: FENCE,
    };
    let frames = extract_catch_up_frames(&spec, fs_plane.as_ref())
        .await
        .expect("the archived stream bridges the fence");
    let rounds: Vec<u64> = frames.iter().map(|f| f.round).collect();
    let expected: Vec<u64> = (FENCE..PRE).collect();
    assert_eq!(
        rounds, expected,
        "the fence-inclusive committed rounds, ascending (defect 20: the fence round rides \
         along for the boot-ambiguous case; a folded one deduplicates guest-side)"
    );
    for f in &frames {
        // The staged frame carries the guest's GENUINE module payload — the canonical
        // VhcMessage::RoundRecord the coordinator authored, not a probe artifact.
        let msg: VhcMessage =
            daemon_vhc_proto::from_canonical_slice(&f.payload).expect("module payload decodes");
        assert!(
            matches!(&msg, VhcMessage::RoundRecord(r) if r.round == f.round),
            "round {} carries its own RoundRecord",
            f.round
        );
    }

    // -- the retention wall: prune one attested segment from the fs plane ------------------------
    let victim = heads.last().expect("heads exist");
    let pruned = seg_root.join(hex_of(&victim.body.segment_hash));
    assert!(pruned.is_file(), "the archived segment object exists");
    std::fs::remove_file(&pruned).expect("retention prunes the object");
    let err = extract_catch_up_frames(&spec, fs_plane.as_ref())
        .await
        .expect_err("a pruned segment is a genuine archive gap");
    assert!(
        matches!(
            &err,
            ReconstructError::Segment { segment, .. } if *segment == victim.body.segment
        ),
        "typed segment refusal naming the missing closure, got {err}"
    );
}

/// **Gate A/E: archive-only recovery under a DIFFERENT base identity.** The two-chain lineage
/// crosses ATTESTORS: chain 1 is attested by the founding box's base key, chain 2 — the
/// successor incarnation on another box — publishes under a SECOND genesis-trusted base
/// identity. After deleting all local run state, a third party holding only the genesis trust
/// set and the product archive reconstructs the full lineage to the recorded head. No key
/// material, no journal, no sidecar of either box survives; the trust anchors are public.
#[tokio::test(flavor = "multi_thread")]
async fn archive_only_recovery_reconstructs_across_a_second_trusted_base_identity() {
    const PRE: u64 = 6;
    const POST: u64 = 2;
    let rig = rig();
    let root = tempdir();
    // The successor box's own base identity — trusted by genesis, distinct from the founder's.
    let base_b = SigningKey::from_bytes(blake3::hash(b"reconstruct/second-base").as_bytes());
    let base_b_id = peer_id(&base_b);
    let trusted = vec![rig.base_id, base_b_id];

    // -- chain 1 (box A): the primary drives PRE rounds, crashes; tail seals; base A attests -----
    let primary_key_seed = *blake3::hash(b"reconstruct/two-bases/primary").as_bytes();
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
    assert!(!heads.is_empty(), "chain 1 attested under base A");

    // -- the successor (box B) reconstructs COLD from the archive and resumes --------------------
    let store: Arc<dyn ContentStore> = segments.clone();
    let capture = reconstruct_coordinator(
        ReconstructSpec {
            heads: heads.clone(),
            run_id: rig.spec.run_id,
            trusted: trusted.clone(),
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
    .expect("box B reconstructs cold from box A's attested archive");
    let standby_key_seed = *blake3::hash(b"reconstruct/two-bases/standby").as_bytes();
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

    // -- chain 2 attests under base B: the lineage now crosses base identities -------------------
    let heads = publish_prefix_of(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &base_b,
        2,
        chain2,
        segments.clone(),
        &heads_dir,
        // Base B's publisher links its founding head to base A's terminal head THROUGH the
        // genesis-trusted set — the seat's recovery lineage survives the box move.
        trusted.clone(),
        Vec::new(),
    )
    .await;
    assert!(
        heads.iter().any(|h| h.body.chain_instance == chain2),
        "chain 2 attested under base B into the shared store"
    );

    // -- DELETE all local run state; a third party recovers on the public trust set alone --------
    let run_state = daemon_vhc_session::journal_home::run_state_dir(&root, RUN_LABEL);
    std::fs::remove_dir_all(&run_state).expect("local run state deleted");
    let store: Arc<dyn ContentStore> = segments.clone();
    let capture2 = reconstruct_coordinator(
        ReconstructSpec {
            heads: heads.clone(),
            run_id: rig.spec.run_id,
            trusted: trusted.clone(),
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
    .expect("the cross-attestor lineage reconstructs from the archive alone");
    let state = coordinator_state_from_capture(&capture2).expect("exported state decodes");
    assert_eq!(
        state.round,
        PRE + POST,
        "the rebuilt seat stands at the recorded head across BOTH base identities"
    );

    // -- certification (matrix case 2): the CROSS-BASE lineage certifies through the kernel ------
    // Same public trust set, same archive; the kernel binds the seam across the attestor change
    // and reports the closure class.
    let report = certify_lineage(
        ReconstructSpec {
            heads,
            run_id: rig.spec.run_id,
            trusted,
            role: "coordinator".into(),
            run_label: RUN_LABEL.into(),
            journal_root: None,
            module: rig.wasm.clone(),
            config: rig.spec.config_bytes.clone(),
            grants: phase_a_grants(),
            incarnation: 4,
            restore: None,
            sidecar_key: None,
            deadline_ms: 60_000,
        },
        segments,
    )
    .await
    .expect("a successor under a different genesis-trusted base certifies GREEN");
    assert_eq!(report.spans.len(), 2, "both incarnation spans replayed");
    assert_eq!(report.seams.len(), 1, "the cross-base seam is bound");
    assert!(
        matches!(report.closure, ClosureClass::Terminal { .. }),
        "the final span's recorded kind-0 terminal closes TERMINAL, got {:?}",
        report.closure
    );
}

/// **Gate E: trainer churn during recovery — the c15b livelock shape through the PRODUCT
/// path.** A trainer whose session churned across the coordinator's crash re-announces the
/// same `Join` it always sends, and it lands on the freshly RECONSTRUCTED coordinator. The
/// resumed seat must serve it as an idempotent catch-up — the retained committed records,
/// ascending, then the standing `RoundOpen` of the active round, byte-identical to the
/// pre-crash flood (frozen at open, never rebuilt from post-churn state) — and the churn must
/// not perturb consensus: the rounds that follow commit byte-identically to an uninterrupted
/// reference. (The tick-level contract is pinned in `tick_lifecycle`; this asserts it holds
/// for a seat that just came back through heads + segments + tail replay.)
#[tokio::test(flavor = "multi_thread")]
async fn a_trainer_rejoin_during_recovery_catches_up_idempotently_on_the_reconstructed_seat() {
    let rig = rig();
    let root = tempdir();
    let pre_kill = build_script(&rig.worker_keys, 0..ROUNDS_BEFORE_KILL);

    // -- the primary drives rounds under the live driver, then crashes ---------------------------
    let primary_key_seed = *blake3::hash(b"reconstruct/churn/primary").as_bytes();
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
    while primary_decisions.len() < decision_count(ROUNDS_BEFORE_KILL, true) {
        let (_, _, msg) = primary
            .next_decision(Duration::from_secs(60))
            .expect("primary decision");
        primary_decisions.push(msg);
    }
    let standing_open = primary_decisions
        .last()
        .cloned()
        .expect("the trailing RoundOpen");
    assert!(
        matches!(&standing_open, VhcMessage::RoundOpen(ro) if ro.round == ROUNDS_BEFORE_KILL),
        "the pre-crash trailing decision is the active round's open"
    );

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
    let end = primary.kill().expect("primary killed");
    assert!(matches!(end, RunEnd::Outcome(_)), "guest thread joined");

    // -- the standby resumes; the churned trainer's re-join lands FIRST --------------------------
    let standby_key_seed = *blake3::hash(b"reconstruct/churn/standby").as_bytes();
    let mut standby = Coordinator::start_migrating(
        &rig.wasm,
        &rig.spec,
        phase_a_grants(),
        1,
        standby_key_seed,
        capture,
    )
    .unwrap();
    let churned = &rig.worker_keys[0];
    standby
        .deliver(
            churned,
            &VhcMessage::Join(Join {
                run_id: RUN_LABEL.into(),
                iroh_id: IrohId([0x44; 32]),
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
                envelope_hash: None,
            }),
        )
        .expect("the churned trainer's re-join delivers");
    // The idempotent catch-up: every retained committed record, ascending, then the standing
    // open — served by the reconstructed seat exactly as a never-crashed one would.
    let mut catch_up = Vec::new();
    for _ in 0..=ROUNDS_BEFORE_KILL {
        let (_, _, msg) = standby
            .next_decision(Duration::from_secs(60))
            .expect("rejoin catch-up decision");
        catch_up.push(msg);
    }
    let replayed: Vec<u64> = catch_up
        .iter()
        .filter_map(|m| match m {
            VhcMessage::RoundRecord(rr) => Some(rr.round),
            _ => None,
        })
        .collect();
    let expected: Vec<u64> = (0..ROUNDS_BEFORE_KILL).collect();
    assert_eq!(
        replayed, expected,
        "the retained committed records replay ascending"
    );
    let reopened = catch_up.last().expect("the standing open closes the flood");
    assert_eq!(
        to_canonical_vec(reopened).unwrap(),
        to_canonical_vec(&standing_open).unwrap(),
        "the standing open re-publishes byte-identical to the pre-crash flood"
    );

    // -- consensus is unperturbed: the following rounds commit byte-identically ------------------
    let post_kill = build_script(
        &rig.worker_keys,
        ROUNDS_BEFORE_KILL..ROUNDS_BEFORE_KILL + ROUNDS_AFTER,
    );
    for sm in &post_kill {
        standby.deliver(&sm.key, &sm.msg).expect("standby deliver");
    }
    let mut standby_decisions = Vec::new();
    for i in 0..decision_count(ROUNDS_AFTER, false) {
        let (_, _, msg) = standby
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("standby decision {i}: {e}"));
        standby_decisions.push(msg);
    }
    standby.stop().expect("standby stops clean");

    // The oracle is an UNINTERRUPTED coordinator fed the identical stream — including the
    // rejoin at the same position (every delivered frame advances the deterministic clock, so
    // the churn is part of the timeline, crashed or not). The reconstructed seat must serve
    // the whole thing byte-identically to the seat that never went down.
    let total_rounds = ROUNDS_BEFORE_KILL + ROUNDS_AFTER;
    let mut reference_script = build_script(&rig.worker_keys, 0..total_rounds);
    reference_script.insert(
        pre_kill.len(),
        ScriptMsg {
            key: churned.clone(),
            msg: VhcMessage::Join(Join {
                run_id: RUN_LABEL.into(),
                iroh_id: IrohId([0x44; 32]),
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
                envelope_hash: None,
            }),
        },
    );
    let catch_up_publishes = ROUNDS_BEFORE_KILL as usize + 1; // the records + the standing open
    let uninterrupted = wasm_reference(
        &rig.wasm,
        &rig.spec,
        primary_key_seed,
        &reference_script,
        decision_count(total_rounds, true) + catch_up_publishes,
    );
    let tail = &uninterrupted[decision_count(ROUNDS_BEFORE_KILL, true)..];
    let resumed: Vec<&VhcMessage> = catch_up.iter().chain(standby_decisions.iter()).collect();
    assert_eq!(tail.len(), resumed.len());
    for (a, b) in tail.iter().zip(resumed.iter()) {
        assert_eq!(
            to_canonical_vec(a).unwrap(),
            to_canonical_vec(b).unwrap(),
            "the reconstructed seat serves mid-recovery churn byte-identically to a seat \
             that never crashed"
        );
    }
}

// == The certification kernel over a SINGLE completed chain (matrix cases 15 and 10) ==============

/// Drive one coordinator to a clean stop under the live driver (optionally through `wrap`, a
/// journaling man-in-the-middle), publish everything through the product publisher, and return
/// the certification inputs. The graceful stop journals the kind-0 terminal, which SEALS the
/// final segment — so the reconciliation sweep archives the complete record including the
/// terminal (the c15l/c15m closure defect's fix, asserted by the TERMINAL closure below).
async fn certified_single_chain(
    rig: &Rig,
    root: &std::path::Path,
    wrap: impl FnOnce(DurableSink) -> Box<dyn JournalSink>,
) -> (
    Vec<daemon_vhc_proto::ArchiveHeadRecord>,
    Arc<MemoryContentStore>,
) {
    const ROUNDS: u64 = 2;
    let key_seed = *blake3::hash(b"reconstruct/certify/single-chain").as_bytes();
    let (sink, chain) = open_live_sink(root, &rig.spec);
    let mut coord = Coordinator::start_with_sink(
        &rig.wasm,
        &rig.spec,
        phase_a_grants(),
        1,
        key_seed,
        wrap(sink),
    )
    .unwrap();
    for sm in &build_script(&rig.worker_keys, 0..ROUNDS) {
        coord.deliver(&sm.key, &sm.msg).expect("deliver");
    }
    for i in 0..decision_count(ROUNDS, false) {
        coord
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("decision {i}: {e}"));
    }
    let end = coord.stop().expect("clean stop");
    assert!(
        matches!(end, RunEnd::Outcome(_)),
        "the guest returned its recorded outcome"
    );
    let segments = Arc::new(MemoryContentStore::new());
    let heads = publish_prefix(
        root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &rig.base_key,
        chain,
        segments.clone(),
        &root.join("heads"),
    )
    .await;
    (heads, segments)
}

fn single_chain_certify_spec(
    rig: &Rig,
    heads: Vec<daemon_vhc_proto::ArchiveHeadRecord>,
) -> ReconstructSpec {
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
        incarnation: 2,
        restore: None,
        sidecar_key: None,
        deadline_ms: 60_000,
    }
}

/// **Matrix case 15 (the single-chain regression): a COMPLETED run's wire archive certifies
/// GREEN through the kernel — the degenerate one-chain lineage — with closure class TERMINAL.**
/// The replay rides the recorded stop into the guest's own `da_run` return and reproduces the
/// recorded outcome; nothing needed exporting past a terminal.
#[tokio::test(flavor = "multi_thread")]
async fn a_completed_single_chain_archive_certifies_terminal_through_the_kernel() {
    let rig = rig();
    let root = tempdir();
    let (heads, segments) = certified_single_chain(&rig, &root, |sink| Box::new(sink)).await;
    let report = certify_lineage(single_chain_certify_spec(&rig, heads), segments)
        .await
        .expect("a completed single-chain archive certifies GREEN");
    assert_eq!(report.spans.len(), 1, "one span, the whole story");
    assert!(report.seams.is_empty(), "no seams in a single chain");
    assert!(
        matches!(report.closure, ClosureClass::Terminal { .. }),
        "a completed run's archive closes TERMINAL, got {:?}",
        report.closure
    );
    assert!(
        report.spans[0].publishes > 0,
        "the decision gate verified the recorded publishes"
    );
}

/// A journaling man-in-the-middle: every record passes through to the durable sink verbatim
/// EXCEPT the first tag-4 publish, whose recorded payload gets one byte flipped (the wire frame
/// and everything else stay authentic). The disk then attests a decision hash the deterministic
/// replay can never re-derive — matrix case 10's mutation, applied at the only layer that can
/// forge it (the journal substrate itself; the content plane is hash-pinned by the heads).
struct TamperFirstPublish {
    inner: DurableSink,
    tampered: bool,
}

impl JournalSink for TamperFirstPublish {
    #[allow(clippy::too_many_arguments)]
    fn run_header(
        &mut self,
        abi: u64,
        worlds: &[(String, u64)],
        bridge: bool,
        manifest: &[u8],
        config: &[u8],
        grants: &[u8],
        resources: RunHeaderResources<'_>,
        channels: &[u8],
        device: &[u8],
    ) -> Result<(), SinkError> {
        self.inner.run_header(
            abi, worlds, bridge, manifest, config, grants, resources, channels, device,
        )
    }
    fn instantiation(&mut self, counter: u64, reason: u64, at: u64) -> Result<(), SinkError> {
        self.inner.instantiation(counter, reason, at)
    }
    fn init(
        &mut self,
        config_hash: [u8; 32],
        grants_hash: [u8; 32],
        status: u64,
    ) -> Result<(), SinkError> {
        self.inner.init(config_hash, grants_hash, status)
    }
    fn execution_grant(
        &mut self,
        execution_grant_hash: [u8; 32],
        status: u64,
    ) -> Result<(), SinkError> {
        self.inner.execution_grant(execution_grant_hash, status)
    }
    fn event(&mut self, at: u64, frame: &[u8]) -> Result<(), SinkError> {
        self.inner.event(at, frame)
    }
    fn signed_frame(
        &mut self,
        channel: u64,
        seq: u64,
        sender: [u8; 32],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        self.inner.signed_frame(channel, seq, sender, frame)
    }
    fn next_seq(&mut self, channel: u64) -> u64 {
        self.inner.next_seq(channel)
    }
    fn publish(
        &mut self,
        channel: u64,
        seq: u64,
        payload: &[u8],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        if !self.tampered && !payload.is_empty() {
            self.tampered = true;
            let mut forged = payload.to_vec();
            forged[0] ^= 0x01;
            return self.inner.publish(channel, seq, &forged, frame);
        }
        self.inner.publish(channel, seq, payload, frame)
    }
    fn clock(&mut self, now: u64) -> Result<(), SinkError> {
        self.inner.clock(now)
    }
    fn timer_arm(&mut self, id: u64, delay: u64, armed_at: u64) -> Result<(), SinkError> {
        self.inner.timer_arm(id, delay, armed_at)
    }
    fn timer_cancel(&mut self, id: u64, status: u64) -> Result<(), SinkError> {
        self.inner.timer_cancel(id, status)
    }
    fn read_back(
        &mut self,
        src: u64,
        kind: u64,
        status: u64,
        value: &[u8],
    ) -> Result<(), SinkError> {
        self.inner.read_back(src, kind, status, value)
    }
    fn device_profile(&mut self, profile: &[u8]) -> Result<(), SinkError> {
        self.inner.device_profile(profile)
    }
    fn drop_coalesced(&mut self, class: u64, rule: u64, dropped: Dropped) -> Result<(), SinkError> {
        self.inner.drop_coalesced(class, rule, dropped)
    }
    fn condition(&mut self, code: &str, detail: &str) -> Result<(), SinkError> {
        self.inner.condition(code, detail)
    }
    fn completion(&mut self, op: u64, result: &[u8]) -> Result<(), SinkError> {
        self.inner.completion(op, result)
    }
    fn snapshot(&mut self, manifest: &[u8]) -> Result<(), SinkError> {
        self.inner.snapshot(manifest)
    }
    fn terminal(
        &mut self,
        kind: u64,
        outcome: Option<u64>,
        trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError> {
        self.inner.terminal(kind, outcome, trap)
    }
}

/// **Matrix case 10: an altered recorded publish refuses as decision divergence.** One byte of
/// one recorded decision payload flipped at journaling time — everything else authentic, every
/// head validly attested, every segment hash-true — and the deterministic replay's own decision
/// gate is what catches the forgery: the replayed publish hash cannot match the recorded one.
#[tokio::test(flavor = "multi_thread")]
async fn an_altered_recorded_publish_refuses_as_decision_divergence() {
    let rig = rig();
    let root = tempdir();
    let (heads, segments) = certified_single_chain(&rig, &root, |sink| {
        Box::new(TamperFirstPublish {
            inner: sink,
            tampered: false,
        })
    })
    .await;
    let err = certify_lineage(single_chain_certify_spec(&rig, heads), segments)
        .await
        .expect_err("a forged decision record must refuse");
    match err {
        ReconstructError::Sandbox(detail) => assert!(
            detail.contains("diverges"),
            "the refusal names the decision divergence: {detail}"
        ),
        other => panic!("expected the decision-gate refusal, got: {other}"),
    }
}
