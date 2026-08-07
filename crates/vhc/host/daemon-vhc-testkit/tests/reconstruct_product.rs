// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The Phase-5 crash-restart regression — the PRODUCT reconstruction path** (§8.8; plan
// "coordinator-reconstruction"): where `failover.rs` proves the ALGORITHM over the dev-side
// oracle journal (`daemon-vhc-observe` records, `AttestedHead`s, testkit chain recovery), this
// drill proves the PRODUCTION machinery end to end:
//
// 1. A primary `coordinator_quorum.wasm` drives rounds; its authoritative inputs are journaled
//    into the SESSION's durable journal home (`DurableSink`) as tag-12 signed-frame records in
//    the live §12.1 wire shape — a sealed prefix plus an unsealed local tail, exactly the disk
//    state a crash leaves.
// 2. The sealed prefix publishes through the PRODUCT archive publisher
//    (`daemon_vhc_session::archive::spawn_archive_publisher`): content-addressed segments on a
//    `ContentStore` + `ArchiveHeadRecord`s attested under a base-issued run-key certificate.
// 3. The primary is killed mid-round-stream.
// 4. `daemon_vhc_session::reconstruct::reconstruct_coordinator` — the executor the worker's
//    join path runs on a `CoordinatorRecovery` directive — re-verifies the heads against the
//    trusted base, recovers the record stream (attested segments + the chained local tail),
//    replays it through a sandboxed instance of the pinned module, and exports the rebuilt
//    state via the typed quiesce→snapshot path.
// 5. A standby boots FROM that capture (`da_migrate`) and resumes; its decisions must be
//    byte-identical to an uninterrupted reference run's — resumption, not approximation.
//
// Also pinned here: the CONFLICTING-HEAD refusal (fork evidence never reconstructs) and the
// untrusted-attestor refusal (a head signed outside the genesis trust never reconstructs).
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

/// Publish the sealed prefix through the PRODUCT publisher and return the attested heads.
async fn publish_prefix(
    root: &std::path::Path,
    run_id: Hash,
    module: Hash,
    base_key: &SigningKey,
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
            instance: 1,
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
            journal_dir: journal_dir(root, RUN_LABEL, "coordinator", 1),
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

#[tokio::test(flavor = "multi_thread")]
async fn crash_restart_reconstructs_through_the_product_path_and_resumes_byte_identically() {
    let rig = rig();
    let root = tempdir();
    let pre_kill = build_script(&rig.worker_keys, 0..ROUNDS_BEFORE_KILL);

    // -- 1. the primary drives rounds; the same inputs land in the durable journal home ----------
    let primary_key_seed = *blake3::hash(b"reconstruct/primary-key").as_bytes();
    let mut primary =
        Coordinator::start(&rig.wasm, &rig.spec, phase_a_grants(), 0, primary_key_seed).unwrap();
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
    // The durable journal a crash leaves: sealed prefix (2/3 of the frames) + unsealed tail.
    let archive_cut = pre_kill.len() * 2 / 3;
    let chain_instance = write_crashed_journal(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &pre_kill,
        archive_cut,
    );
    // The fixture's on-disk shape IS the §8 crash contract — pin it before reconstructing.
    {
        let jdir = journal_dir(&root, RUN_LABEL, "coordinator", 1);
        let paths = daemon_vhc_journal::JournalPaths::open(&jdir).expect("journal home");
        let frames_of = |ord: u64| {
            let scan = daemon_vhc_journal::scan_file(paths.segment(ord)).expect("scan");
            let frames = scan
                .records
                .iter()
                .filter(|r| {
                    matches!(&r.body,
                        daemon_vhc_journal::Body::SignedFrame(sf) if sf.frame.is_some())
                })
                .count();
            (scan.sealed, frames)
        };
        assert_eq!(
            frames_of(0),
            (true, archive_cut),
            "the sealed prefix carries the pre-cut tag-12 frames inline"
        );
        assert_eq!(
            frames_of(1),
            (false, pre_kill.len() - archive_cut),
            "the unsealed local tail carries the post-cut tag-12 frames inline"
        );
    }

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

    // -- 3. kill the primary mid-stream -----------------------------------------------------------
    let end = primary.kill().expect("primary killed");
    assert!(matches!(end, RunEnd::Outcome(_)), "guest thread joined");

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

/// The c15-20260806b/g regression: a coordinator whose config enables availability
/// verification (coordinator-as-storage-client, §6.4 I6 — the fleet ceremony's shape) issues
/// one `payload_get` per replayed Commitment. The reconstruction sandbox services no
/// capability providers; ops are completed promptly as typed failures AND the sandbox lifts
/// the `max_outstanding` ceiling (`max_outstanding_ops = 0`), because prompt completion
/// alone is a scheduling race: the host-side drain is polled, while the guest folds the
/// spooled lineage synchronously and can burst one `payload_get` per Commitment past any
/// poll cadence (c15-20260806b: the un-serviced queue crossed 16; c15-20260806g head 12:
/// the fold burst outran the 2ms drain and trapped `GrantViolation` on every retry until
/// the run went terminal). Eighty commitments (40 rounds x 2 workers) cross the default
/// ceiling many times over; the reconstruction must still complete and stand at the next
/// un-opened round.
#[tokio::test(flavor = "multi_thread")]
async fn availability_checked_lineage_reconstructs_past_the_op_ceiling() {
    const ROUNDS: u64 = 40;
    let rig = rig();
    let config = with_verify_availability(&rig.spec.config_bytes);
    let root = tempdir();
    let script = build_script(&rig.worker_keys, 0..ROUNDS);
    let chain_instance = write_crashed_journal(
        &root,
        rig.spec.run_id,
        rig.spec.module_hash,
        &script,
        script.len() * 2 / 3,
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
            journal_root: Some(root),
            module: rig.wasm.clone(),
            config,
            grants: phase_a_grants(),
            incarnation: 1,
            restore: None,
            deadline_ms: 60_000,
        },
        store,
    )
    .await
    .expect("an availability-checking lineage reconstructs past the op ceiling");
    let state = coordinator_state_from_capture(&capture).expect("exported state decodes");
    assert_eq!(
        state.round, ROUNDS,
        "the reconstructed state stands at the next un-opened round"
    );
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
