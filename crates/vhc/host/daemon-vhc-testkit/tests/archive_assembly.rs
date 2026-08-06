// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The product-archive → GREEN replay gate** (the Phase 4 acceptance): a real sandboxed
// coordinator run is journaled and published INCREMENTALLY through the PRODUCT path — the
// journal's per-seal hook feeding `spawn_archive_publisher` into the filesystem archive-head +
// content stores (ABI §8.8) — then a third party holding ONLY those stores assembles the §3.4
// replay layout (`assemble_archive`) and re-verifies the whole run through the sandboxed
// consensus oracle. Nothing is hand-attested: every head is authored by the publisher, every
// byte crosses the untrusted stores, every verification is the reader's own.
//
// Dev/test harness: shells `cargo build` for guests; journal + stores write real files.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ciborium::value::Value;

use daemon_vhc_net::transport::ContentStore;
use daemon_vhc_net::{ArchiveHeadStore, FsArchiveHeadStore, FsContentStore};
use daemon_vhc_observe::journal::archive::ChainHead;
use daemon_vhc_observe::journal::oracle::{record_initial_state, record_input, record_run_header};
use daemon_vhc_observe::journal::record::ExecIdentity;
use daemon_vhc_observe::journal::store::{Journal, RotatePolicy};
use daemon_vhc_observe::journal::StaticKey;
use daemon_vhc_observe::{
    assemble_archive, coordinator_lineage, envelope_trusted_bases,
    replay_consensus_from_verified_archive, verify_chains, RecordArchive, ReplicationPolicy,
    RetentionPolicy,
};
use daemon_vhc_proto::archive::ArchiveHeadRecord;
use daemon_vhc_proto::cert::{CertScope, RunKeyCertificate};
use daemon_vhc_proto::genesis::GenesisEnvelope;
use daemon_vhc_proto::{
    blake3_hash, from_canonical_slice, peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId,
    PeerId, SigningKey, StateDigest, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::Input;
use daemon_vhc_sdk_consensus::messages::{
    Commitment, Digest, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass,
};
use daemon_vhc_sdk_consensus::{SignedMessage, VhcMessage};
use daemon_vhc_session::archive::{spawn_archive_publisher, ArchiveSpec, SignerBinding};
use daemon_vhc_session::replay_sandbox::SandboxedCoordinator;
use daemon_vhc_testkit::genesis_run::{phase_a_grants, EnvelopeInputs};
use daemon_vhc_testkit::live_genesis::fixture_authored_execution;
use daemon_vhc_testkit::{configure_coordinator, genesis_envelope, Coordinator};

const ROUNDS: u64 = 4;
const RUN_LABEL: &str = "archive-assembly";

fn coordinator_quorum_wasm() -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm("coordinator_quorum")
}

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    base.push(format!(
        "dvhc-archive-assembly-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn sign(k: &SigningKey, m: &VhcMessage) -> SignedMessage {
    SignedMessage::sign(k, VHC_PROTO_VERSION, m.clone()).expect("sign")
}

/// The scripted run: join/warmup, then per round a per-worker commitment (+ its content-plane
/// payload bytes), a per-worker post-ingest state digest, and the covering storage receipt.
struct Script {
    msgs: Vec<(SigningKey, VhcMessage)>,
    /// hash → payload bytes (what the workers put on the content plane).
    payloads: BTreeMap<Hash, Vec<u8>>,
}

fn build_script(worker_keys: &[SigningKey; 2]) -> Script {
    let peers: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();
    let mut msgs = Vec::new();
    let mut payloads = BTreeMap::new();
    for k in worker_keys {
        msgs.push((
            k.clone(),
            VhcMessage::Join(Join {
                run_id: RUN_LABEL.into(),
                iroh_id: IrohId([0x44; 32]),
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
                envelope_hash: None,
            }),
        ));
    }
    for k in worker_keys {
        msgs.push((
            k.clone(),
            VhcMessage::Heartbeat(Heartbeat {
                round: 0,
                ready: Some(true),
            }),
        ));
    }
    for round in 0..ROUNDS {
        let mut entries = Vec::new();
        for (i, k) in worker_keys.iter().enumerate() {
            let bytes = format!("update/{i}/{round}").into_bytes();
            let hash = blake3_hash(&bytes);
            payloads.insert(hash, bytes.clone());
            msgs.push((
                k.clone(),
                VhcMessage::Commitment(Commitment {
                    round,
                    payload: hash,
                    size: bytes.len() as u64,
                    locators: Vec::new(),
                }),
            ));
            entries.push(RecordEntry {
                peer: peers[i],
                hash,
                size: bytes.len() as u64,
            });
        }
        // Replicas that ingested the same committed set announce the SAME digest — the per-peer
        // agreement oracle in `vhc-replay` verifies exactly that.
        for k in worker_keys {
            msgs.push((
                k.clone(),
                VhcMessage::Digest(Digest {
                    round,
                    digest: StateDigest([u8::try_from(round % 251).unwrap() + 1; 16]),
                }),
            ));
        }
        msgs.push((
            worker_keys[0].clone(),
            VhcMessage::StorageReceipt(StorageReceipt {
                round,
                verified: entries,
            }),
        ));
    }
    Script { msgs, payloads }
}

#[test]
#[allow(clippy::too_many_lines)]
fn product_archive_assembles_and_replays_green() {
    let wasm = coordinator_quorum_wasm();
    let coord_hash = Hash(*blake3::hash(&wasm).as_bytes());

    // Identities (D1 cert layering): the envelope-named base identity signs certificates; the
    // coordinator session's PER-RUN key signs its publishes and its archive heads.
    let base_key = SigningKey::from_bytes(blake3::hash(b"assembly/base-identity").as_bytes());
    let base_id = peer_id(&base_key);
    let run_key_seed = *blake3::hash(b"assembly/run-key").as_bytes();
    let run_key = SigningKey::from_bytes(&run_key_seed);
    let worker_keys = [
        SigningKey::from_bytes(blake3::hash(b"assembly/worker/0").as_bytes()),
        SigningKey::from_bytes(blake3::hash(b"assembly/worker/1").as_bytes()),
    ];

    // The frozen genesis: its canonical bytes ARE the envelope.cbor the assembler verifies from,
    // and their blake3 is the run id every head must name.
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
    let author = SigningKey::from_bytes(blake3::hash(b"assembly/author").as_bytes());
    let frozen = genesis.freeze(&author).expect("genesis freeze");
    let spec = configure_coordinator(&frozen).expect("coordinator configurable");
    let run_id = *frozen.run_id();
    assert_eq!(run_id, blake3_hash(frozen.bytes()), "run id IS the bytes");

    let certificate = RunKeyCertificate::issue(
        &base_key,
        CertScope {
            run_id,
            epoch: 0,
            role: "coordinator".into(),
            instance: 0,
            module_hash: coord_hash,
        },
        peer_id(&run_key),
    )
    .expect("issue run-key certificate");

    // -- the PRODUCT stores + publisher (fs planes; the acceptance-baseline configuration) -------
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let stores_root = tempdir();
    let heads_store: Arc<dyn ArchiveHeadStore> = Arc::new(
        rt.block_on(FsArchiveHeadStore::open(&stores_root.join("heads")))
            .expect("open head store"),
    );
    let content_store =
        Arc::new(FsContentStore::open(&stores_root.join("content")).expect("open content store"));

    // The journal, armed with the per-seal hook — the product wiring the durable journal home
    // performs (`DurableSink::arm_seal_hook`).
    let ident = ExecIdentity {
        run_id,
        epoch: 0,
        role: "coordinator".into(),
        instance: 0,
        module: coord_hash,
    };
    let jroot = tempdir();
    let mut journal = Journal::create(
        &jroot,
        ident.clone(),
        StaticKey::new([7u8; 32]),
        // A small threshold: the run spans several sealed segments, each published per seal.
        RotatePolicy {
            max_records: 8,
            ..RotatePolicy::default()
        },
    )
    .expect("journal create");
    let (seal_tx, seal_rx) = tokio::sync::mpsc::unbounded_channel();
    journal.set_seal_hook(Box::new(move |sealed| {
        let _ = seal_tx.send(sealed.clone());
    }));

    let bindings = Arc::new(Mutex::new(vec![SignerBinding {
        signing_seed: run_key_seed,
        certificate,
    }]));
    let publisher = {
        let _guard = rt.enter();
        spawn_archive_publisher(
            RUN_LABEL.to_string(),
            run_id,
            "coordinator".to_string(),
            ArchiveSpec {
                seals: seal_rx,
                journal_dir: jroot.clone(),
                chain_instance: 0,
                round_claim: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            },
            Arc::clone(&heads_store),
            content_store.clone(),
            bindings,
        )
    };

    // -- drive the real sandboxed coordinator run, journaling inputs + publishes -----------------
    record_run_header(&mut journal, &ident, Vec::new()).expect("run header");
    let initial = {
        // The genesis config's `{state: …}` opaque map carries the initial coordinator state.
        let v: Value =
            ciborium::de::from_reader(spec.config_bytes.as_slice()).expect("config cbor");
        let Value::Map(entries) = v else {
            panic!("coordinator config is a map")
        };
        entries
            .iter()
            .find_map(|(k, val)| match k {
                Value::Text(t) if t == "state" => Some(val.clone()),
                _ => None,
            })
            .expect("config carries state")
            .deserialized()
            .expect("state decodes")
    };
    record_initial_state(&mut journal, &initial).expect("snapshot");

    let script = build_script(&worker_keys);
    let mut coord = Coordinator::start(&wasm, &spec, phase_a_grants(), 0, run_key_seed).unwrap();
    for (at, (key, msg)) in script.msgs.iter().enumerate() {
        coord.deliver(key, msg).expect("deliver");
        record_input(&mut journal, at as u64, &Input::Message(sign(key, msg)))
            .expect("record input");
    }
    // Drain the published decisions (per committed round: RoundOpen + RoundRecord, plus the
    // trailing open of the never-recorded next round), journaling each tag-4 as the session does
    // — signed by the run key, the frame the archive carries.
    let decisions = 2 * usize::try_from(ROUNDS).unwrap() + 1;
    for _ in 0..decisions {
        let (_, _, msg) = coord
            .next_decision(Duration::from_secs(60))
            .expect("decision");
        let payload = to_canonical_vec(&msg).expect("payload cbor");
        let signed = SignedMessage::sign(&run_key, VHC_PROTO_VERSION, msg).expect("sign publish");
        let frame = to_canonical_vec(&signed).expect("frame cbor");
        journal
            .publish(0, &payload, frame)
            .expect("journal publish");
    }
    coord.stop().expect("clean stop");
    journal.commit().expect("final barrier");
    journal.roll().expect("terminal seal");
    drop(journal); // closes the seal stream; the publisher drains and exits

    rt.block_on(publisher).expect("publisher drained clean");

    // -- the workers' payloads + the genesis-pinned module ride the content plane ----------------
    rt.block_on(async {
        for bytes in script.payloads.values() {
            content_store.put_content(bytes).await.expect("put payload");
        }
        content_store.put_content(&wasm).await.expect("put module");
    });

    // -- a third party assembles the §3.4 layout from the untrusted stores alone -----------------
    let published = rt.block_on(heads_store.fetch_heads()).expect("fetch heads");
    assert!(
        published.len() >= 3,
        "the run spans several per-seal published heads (got {})",
        published.len()
    );
    // `DVHC_KEEP_ARCHIVE=<dir>` keeps the assembled layout for a manual `xtask vhc-replay`
    // smoke over the same product archive (the runbook's local verification affordance).
    let out = std::env::var_os("DVHC_KEEP_ARCHIVE").map_or_else(tempdir, std::path::PathBuf::from);
    println!(
        "assembled archive: {} (run {})",
        out.display(),
        run_id.to_hex()
    );
    let mut fetch = |hash: &Hash| -> Result<Vec<u8>, String> {
        rt.block_on(content_store.get_content(hash))
            .map_err(|e| e.to_string())
    };
    // The registry serves the SIGNED wire form (`SignedEnvelope { bytes, signature, signer }`);
    // the assembler must verify + unwrap it to the frozen inner bytes (the c15d pull failed
    // exactly here — the wire form was decoded directly as the inner envelope). Assemble from
    // the wire form (the production shape), then re-assemble from the bare inner bytes into a
    // second layout and hold both reports to the same verdict (the back-compat seam).
    let wire = to_canonical_vec(&daemon_vhc_proto::SignedEnvelope {
        bytes: frozen.bytes().to_vec(),
        signature: *frozen.signature(),
        signer: *frozen.signer(),
    })
    .expect("encode signed envelope wire");
    let report = assemble_archive(&out, &wire, published.clone(), &mut fetch)
        .expect("assembly verifies + writes the layout from the signed wire form");
    let inner_out = tempdir();
    let inner_report = assemble_archive(&inner_out, frozen.bytes(), published, &mut fetch)
        .expect("assembly verifies + writes the layout from the bare inner bytes");
    assert_eq!(inner_report.run_id, report.run_id);
    assert_eq!(inner_report.chains_verified, report.chains_verified);
    assert_eq!(inner_report.payloads_written, report.payloads_written);
    assert_eq!(
        std::fs::read(out.join("envelope.cbor")).expect("wire-form layout envelope"),
        frozen.bytes(),
        "the layout carries the frozen INNER bytes whichever form arrived"
    );
    assert_eq!(report.run_id, run_id);
    assert_eq!(report.chains_verified, 1);
    assert_eq!(report.coordinator_lineage, vec![0]);
    assert_eq!(
        report.payloads_written,
        script.payloads.len() as u64,
        "every committed payload assembled"
    );
    assert_eq!(
        report.peer_transcripts, 2,
        "both workers' digest transcripts"
    );

    // -- and re-verifies the run through the consensus oracle from the layout alone --------------
    let envelope_bytes = std::fs::read(out.join("envelope.cbor")).expect("read envelope");
    let envelope: GenesisEnvelope = from_canonical_slice(&envelope_bytes).expect("decode envelope");
    let trusted = envelope_trusted_bases(&envelope);
    let heads_bytes = std::fs::read(out.join("heads.cbor")).expect("read heads");
    let records: Vec<ArchiveHeadRecord> = from_canonical_slice(&heads_bytes).expect("decode heads");
    let chains = verify_chains(&run_id, &trusted, records).expect("chains verify");
    let lineage = coordinator_lineage(&chains, "coordinator").expect("lineage");
    let heads: Vec<ChainHead> = lineage[0]
        .heads
        .iter()
        .map(|r| ChainHead {
            run_id: r.body.run_id,
            epoch: r.body.epoch,
            role: r.body.role.clone(),
            instance: r.body.instance,
            module: r.body.module,
            segment: r.body.segment,
            segment_hash: r.body.segment_hash,
            prev_hash: r.body.prev_hash,
            records: r.body.records,
        })
        .collect();

    let mut archive = RecordArchive::new(
        spec.authority.clone(),
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    for entry in std::fs::read_dir(out.join("segments")).expect("segments dir") {
        let bytes = std::fs::read(entry.expect("entry").path()).expect("read segment");
        archive.publish_segment(bytes).expect("publish segment");
    }
    let mut payloads: BTreeMap<Hash, Vec<u8>> = BTreeMap::new();
    for entry in std::fs::read_dir(out.join("payloads")).expect("payloads dir") {
        let bytes = std::fs::read(entry.expect("entry").path()).expect("read payload");
        payloads.insert(blake3_hash(&bytes), bytes);
    }

    let sandbox = SandboxedCoordinator::new(
        std::fs::read(out.join("coordinator.wasm")).expect("read module"),
    );
    let verdict = replay_consensus_from_verified_archive(&sandbox, &archive, &heads, &payloads)
        .expect("consensus replay GREEN");
    assert_eq!(verdict.replay.rounds_verified, ROUNDS);
    assert_eq!(verdict.set_commitments_verified, ROUNDS);
    assert_eq!(verdict.payload_entries_verified, ROUNDS * 2);
    assert_eq!(verdict.segments_verified as usize, heads.len());
}
