// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The adversarial `Authority` suite** (refactor §8/D2 acceptance; architecture §4.2: "the
// conformance surface for an `Authority` is adversarial, not merely functional — partitions,
// equivocation, withheld records, and conflicting valid-looking histories are standing test
// cases, run in the simulator against the production module blob").
//
// Every lane here drives the PRODUCTION `coordinator_quorum.wasm` blob (or verifies its archived
// decisions) and judges authority exclusively through D1's contract (`AuthorityConfig::authorize`
// / the typed `AuthError`s):
//
// 1. **Equivocation → portable evidence, end-to-end**: one SingleKey custody drives TWO blob
//    instances of the same run (the duplicated/leaked-key scenario) on divergent inputs; each
//    side's archived history is INDIVIDUALLY green under consensus replay ("conflicting
//    valid-looking histories"), and the two attested heads at the same height are self-contained
//    `DivergentHead` evidence a fresh third party verifies from the heads + the run's declared
//    `AuthorityConfig` alone (architecture §10 detectability).
// 2. **Withheld records**: a worker's commitments are withheld from the blob; rounds close by the
//    (deterministic, event-count-clock) deadline WITHOUT the withheld peer, absences accrue, and
//    the peer is dropped at `k_absences` — byte-identical to the native reference, liveness
//    preserved. The host layer's withholding counterpart (SpoolFull back-pressure, never a silent
//    drop) is pinned by the pump-hold rig (`pump_backpressure.rs`).
// 3. **Partition (eclipse) heal**: a peer that observed NOTHING of the live run reconstructs and
//    re-verifies the whole history from the record archive + payloads alone and converges to the
//    live coordinator's exact final state (the §10 fork-evidence availability assumption's happy
//    half; the divergent-histories half is lane 1).
// 4. **ThresholdKeys (m-of-n) quorum refusals, end-to-end**: heads under a 2-of-3 topology —
//    quorum-attested accepted through archive + consensus replay; sub-quorum / duplicate-signer /
//    non-member / bad-signature / single-transport-signature all refused with D1's typed
//    `AuthError`s; and two quorum-attested divergent heads still yield portable fork evidence
//    (quorum intersection pins at least one faulty signer).
//
// Dev/test harness: shells `cargo build` for guests; journal writes real files.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::time::Duration;

use ciborium::value::Value;

use daemon_vhc_observe::journal::archive::{AttestedHead, ChainHead};
use daemon_vhc_observe::journal::oracle::{record_initial_state, record_input, record_run_header};
use daemon_vhc_observe::journal::record::ExecIdentity;
use daemon_vhc_observe::journal::segment::scan_file;
use daemon_vhc_observe::journal::store::{Journal, RotatePolicy};
use daemon_vhc_observe::journal::StaticKey;
use daemon_vhc_observe::{
    detect_fork, replay_consensus_from_archive, ArchiveError, ForkEvidence, RecordArchive,
    ReplicationPolicy, RetentionPolicy,
};
// The consensus-replay oracle re-derives inside the real coordinator module : the tests hold
// the same `coordinator_quorum.wasm` blob they drove, so they replay through it.
use daemon_vhc_proto::envelope::{GlobalBatch, StopCondition};
use daemon_vhc_proto::{
    blake3_hash, peer_id, sign_canonical, to_canonical_vec, CapabilitySet, Hash, IrohId, PeerId,
    Seed, SigningKey, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, Input, RunConfig};
use daemon_vhc_sdk_consensus::messages::{
    Commitment, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass,
};
use daemon_vhc_sdk_consensus::{
    AuthError, AuthorityConfig, RecordSig, SingleKey, ThresholdKeys, Topology,
    DEFAULT_RECORDS_CHANNEL,
};
use daemon_vhc_sdk_consensus::{SignedMessage, VhcMessage};
use daemon_vhc_session::replay_sandbox::SandboxedCoordinator;
use daemon_vhc_testkit::genesis_run::phase_a_grants;
use daemon_vhc_testkit::{Coordinator, CoordinatorSpec};

// -- guest build (the established testkit pattern) -------------------------------------------------

fn coordinator_quorum_wasm() -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm("coordinator_quorum")
}

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    base.push(format!(
        "dvhc-adversarial-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

// -- shared drive plumbing --------------------------------------------------------------------------

struct ScriptMsg {
    key: SigningKey,
    msg: VhcMessage,
}

fn sign(k: &SigningKey, m: &VhcMessage) -> SignedMessage {
    SignedMessage::sign(k, VHC_PROTO_VERSION, m.clone()).expect("sign")
}

fn single_key(authority: PeerId) -> AuthorityConfig {
    AuthorityConfig {
        topology: Topology::SingleKey(SingleKey::new(authority)),
        records_channel: DEFAULT_RECORDS_CHANNEL,
    }
}

/// A coordinator run config with adversarial-lane knobs: deadlines the event-count clock makes
/// deterministic, and a small `k_absences` for the withholding lane.
fn run_config(
    run_label: &str,
    min_peers: u32,
    round_train_max_s: u64,
    k_absences: u32,
) -> RunConfig {
    RunConfig {
        run_id: run_label.to_string(),
        proto_version: VHC_PROTO_VERSION,
        envelope_hash: Hash([0u8; 32]),
        required_capabilities: CapabilitySet::new(),
        min_peers,
        max_peers: 4,
        warmup_s: 1_000_000,
        round_train_max_s,
        round_witness_s: 1_000_000,
        cooldown_s: 1_000_000,
        epoch_rounds: 0,
        stall_rounds_max: 2,
        global_batch: GlobalBatch {
            start: 4,
            end: 4,
            ramp_rounds: 1,
        },
        stop: StopCondition::Rounds(1_000_000),
        steps_per_round: 2,
        seq_len: 9,
        witness_target: 0,
        overlap_bps: 0,
        k_absences,
        verification_percent: 0,
        authorized: Vec::new(),
    }
}

/// The blob spec for a directly-authored coordinator state (the spec seat is public; the
/// genesis-configured path is pinned by the end-state/matrix lanes).
fn spec_for(
    wasm: &[u8],
    state: &CoordinatorState,
    authority: AuthorityConfig,
    run_id: Hash,
) -> CoordinatorSpec {
    let config_bytes = {
        let v = Value::Map(vec![(
            Value::Text("state".into()),
            Value::serialized(state).expect("state value"),
        )]);
        to_canonical_vec(&v).expect("config cbor")
    };
    CoordinatorSpec {
        module_hash: Hash(*blake3::hash(wasm).as_bytes()),
        config_bytes,
        authority,
        run_id,
    }
}

/// Drive the production blob over `script`, journaling every input (msg + clock pair) AND every
/// blob decision (tag-4, custody-signed) into an on-disk journal; seal + archive everything under
/// heads attested by `custody`. Returns the archive, its heads, and the blob's decisions.
///
/// The drive drains until the blob has published `expected_records` `RoundRecord`s — the blob IS
/// the oracle (consensus never runs outside the sandboxed module; the D2 dual-compilation gate is
/// the identity proof, so no separate native fold is authored here).
fn drive_and_archive(
    wasm: &[u8],
    spec: &CoordinatorSpec,
    key_seed: [u8; 32],
    custody: &SigningKey,
    script: &[ScriptMsg],
    expected_records: usize,
) -> (RecordArchive, Vec<AttestedHead>, Vec<VhcMessage>) {
    let ident = ExecIdentity {
        run_id: spec.run_id,
        epoch: 0,
        role: "coordinator".into(),
        instance: 0,
        module: spec.module_hash,
    };
    let root = tempdir();
    let mut journal = Journal::create(
        &root,
        ident.clone(),
        StaticKey::new([7u8; 32]),
        RotatePolicy {
            max_records: 10_000,
            ..RotatePolicy::default()
        },
    )
    .expect("journal create");
    record_run_header(&mut journal, &ident, Vec::new()).expect("run header");
    let initial = {
        // Decode the initial state back out of the spec's config bytes (one authoring source).
        let v: Value = ciborium::de::from_reader(spec.config_bytes.as_slice()).expect("cfg");
        let Value::Map(entries) = v else {
            panic!("cfg map")
        };
        entries
            .iter()
            .find_map(|(k, val)| match k {
                Value::Text(t) if t == "state" => Some(val.clone()),
                _ => None,
            })
            .expect("state")
            .deserialized::<CoordinatorState>()
            .expect("state decodes")
    };
    record_initial_state(&mut journal, &initial).expect("snapshot");

    let mut coord = Coordinator::start(wasm, spec, phase_a_grants(), 0, key_seed).unwrap();
    let mut at = 0u64;
    let mut now_s = initial.now_s;
    for sm in script {
        coord.deliver(&sm.key, &sm.msg).expect("deliver");
        record_input(&mut journal, at, &Input::Message(sign(&sm.key, &sm.msg))).expect("msg");
        at += 1;
        now_s += 1;
        record_input(&mut journal, at, &Input::Clock(now_s)).expect("clock");
        at += 1;
    }
    let mut decisions = Vec::new();
    let mut records = 0usize;
    while records < expected_records {
        let (_, _, msg) = coord
            .next_decision(Duration::from_secs(60))
            .expect("blob decision");
        if matches!(msg, VhcMessage::RoundRecord(_)) {
            records += 1;
        }
        // Journal the blob's decision as the custody-signed tag-4 publish (the oracle).
        let payload = to_canonical_vec(&msg).expect("payload");
        let signed = sign(custody, &msg);
        let frame = to_canonical_vec(&signed).expect("frame");
        journal
            .publish(0, &payload, frame)
            .expect("journal publish");
        decisions.push(msg);
    }
    coord.stop().expect("blob stops clean");
    journal.roll().expect("seal");

    let mut archive = RecordArchive::new(
        spec.authority.clone(),
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    let mut heads = Vec::new();
    let mut prev = Hash([0u8; 32]);
    let ords = journal.paths().existing_segments().expect("segments");
    for &ord in &ords {
        let scan = scan_file(journal.paths().segment(ord)).expect("scan");
        if !scan.sealed {
            continue;
        }
        let bytes = std::fs::read(journal.paths().segment(ord)).expect("read");
        let addr = archive.publish_segment(bytes).expect("publish");
        let head = AttestedHead::single(
            custody,
            ChainHead {
                run_id: ident.run_id,
                epoch: 0,
                role: "coordinator".into(),
                instance: 0,
                module: ident.module,
                segment: ord,
                segment_hash: addr,
                prev_hash: prev,
                // The seal record is scanned but not counted — the head attests the seal's own
                // declared count (binding.rs: `seal.records` excludes the seal frame itself).
                records: scan.records.len() as u64 - 1,
            },
        )
        .expect("attest");
        // Under a SingleKey topology the head ingests here; a ThresholdKeys caller re-attests
        // with its quorum (lane 4) — a single custody signature is correctly sub-quorum there.
        if archive.head_is_authoritative(&head) {
            archive.ingest_head(head.clone()).expect("accepted");
        }
        heads.push(head);
        prev = addr;
    }
    (archive, heads, decisions)
}

/// The prologue + per-round commit/receipt script for `worker_keys`, with `marker`-distinguished
/// payload content per round (the divergence knob for the equivocation lane).
fn barrier_script(
    run_label: &str,
    worker_keys: &[SigningKey],
    rounds: std::ops::Range<u64>,
    marker: &str,
    payloads: &mut BTreeMap<Hash, Vec<u8>>,
) -> Vec<ScriptMsg> {
    let peers: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();
    let mut script = Vec::new();
    if rounds.start == 0 {
        for k in worker_keys {
            script.push(ScriptMsg {
                key: k.clone(),
                msg: VhcMessage::Join(Join {
                    run_id: run_label.into(),
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
            let bytes = format!("update/{marker}/{i}/{round}").into_bytes();
            let hash = blake3_hash(&bytes);
            payloads.insert(hash, bytes.clone());
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

// -- lane 1: equivocation → portable evidence, end-to-end -------------------------------------------

#[test]
fn equivocation_yields_portable_evidence_and_both_histories_look_valid() {
    let wasm = coordinator_quorum_wasm();
    // ONE custody key (the duplicated/leaked SingleKey scenario) drives TWO instances of the
    // same run on divergent inputs.
    let custody = SigningKey::from_bytes(blake3::hash(b"equiv/custody").as_bytes());
    let authority = single_key(peer_id(&custody));
    let run_id = Hash(*blake3::hash(b"equiv/run").as_bytes());
    let worker_keys = [
        SigningKey::from_bytes(blake3::hash(b"equiv/w0").as_bytes()),
        SigningKey::from_bytes(blake3::hash(b"equiv/w1").as_bytes()),
    ];
    let initial = CoordinatorState::new(run_config("equiv", 2, 1_000_000, 8), Seed([0x33; 32]), 0);
    let spec = spec_for(&wasm, &initial, authority.clone(), run_id);
    let key_seed = *blake3::hash(b"equiv/frame-key").as_bytes();

    // Side A and side B: identical prologue, DIVERGENT round payloads.
    let mut payloads_a = BTreeMap::new();
    let script_a = barrier_script("equiv", &worker_keys, 0..2, "A", &mut payloads_a);
    let mut payloads_b = BTreeMap::new();
    let script_b = barrier_script("equiv", &worker_keys, 0..2, "B", &mut payloads_b);

    // Each side drives 2 barrier rounds to a record; the blob is the oracle (drain by record).
    let (archive_a, heads_a, _) = drive_and_archive(&wasm, &spec, key_seed, &custody, &script_a, 2);
    let (archive_b, heads_b, _) = drive_and_archive(&wasm, &spec, key_seed, &custody, &script_b, 2);

    // "Conflicting valid-looking histories": EACH side is green in isolation — a third party
    // shown only one archive finds nothing wrong (consensus replay re-derives every record and
    // digest from archive + payloads alone).
    let sandbox = SandboxedCoordinator::new(wasm.clone());
    let a = replay_consensus_from_archive(&sandbox, &archive_a, &heads_a, &payloads_a)
        .expect("A valid");
    let b = replay_consensus_from_archive(&sandbox, &archive_b, &heads_b, &payloads_b)
        .expect("B valid");
    assert_eq!(a.replay.rounds_verified, 2);
    assert_eq!(b.replay.rounds_verified, 2);
    assert_ne!(
        a.replay.final_state_hash, b.replay.final_state_hash,
        "the two histories genuinely diverge"
    );

    // The moment the heads MEET, the fork is exposed: same height, same custody, different
    // content — self-contained portable evidence (architecture §4.3/§10).
    let evidence = detect_fork(&[heads_a[0].clone(), heads_b[0].clone()], &authority)
        .expect("divergent heads at height 0");
    let ForkEvidence::DivergentHead { a: ha, b: hb } = &evidence else {
        panic!("expected DivergentHead, got {evidence:?}");
    };
    // PORTABILITY: a FRESH third party holding nothing but the two heads and the run's declared
    // AuthorityConfig re-verifies the evidence — both heads authorize standalone, same scope and
    // height, different content. Nothing else is needed.
    let fresh = single_key(peer_id(&custody));
    for h in [ha.as_ref(), hb.as_ref()] {
        fresh
            .authorize(&h.preimage().expect("preimage"), &h.sigs)
            .expect("head authorizes standalone");
    }
    assert_eq!(ha.body.segment, hb.body.segment);
    assert_eq!(ha.body.run_id, hb.body.run_id);
    assert_ne!(ha.body.segment_hash, hb.body.segment_hash);

    // The archive's live ingest path yields the same evidence (side A accepted, side B's head
    // conflicts at the same height).
    let mut third_party = RecordArchive::new(
        authority,
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    assert_eq!(third_party.ingest_head(heads_a[0].clone()).unwrap(), None);
    assert!(matches!(
        third_party.ingest_head(heads_b[0].clone()).unwrap(),
        Some(ForkEvidence::DivergentHead { .. })
    ));
}

// -- lane 2: withheld records (deterministic deadline close + absence drop, blob ≡ native) ---------

#[test]
fn withheld_records_close_rounds_without_the_peer_and_drop_it_at_k() {
    let wasm = coordinator_quorum_wasm();
    let custody = SigningKey::from_bytes(blake3::hash(b"withhold/custody").as_bytes());
    let authority = single_key(peer_id(&custody));
    let run_id = Hash(*blake3::hash(b"withhold/run").as_bytes());
    let w0 = SigningKey::from_bytes(blake3::hash(b"withhold/w0").as_bytes());
    let w1 = SigningKey::from_bytes(blake3::hash(b"withhold/w1").as_bytes());
    let withheld_peer = peer_id(&w1);
    // Tiny round deadline (deterministic under the event-count clock); k_absences 2.
    let initial = CoordinatorState::new(run_config("withhold", 2, 5, 2), Seed([0x33; 32]), 0);
    let spec = spec_for(&wasm, &initial, authority, run_id);
    let key_seed = *blake3::hash(b"withhold/frame-key").as_bytes();

    // The script: both join + ready; w1's commitments are WITHHELD every round. Each round: w0's
    // commitment + a receipt covering w0 only + filler heartbeats that tick the deterministic
    // event-count clock past the deadline.
    let mut payloads = BTreeMap::new();
    let mut script = Vec::new();
    for k in [&w0, &w1] {
        script.push(ScriptMsg {
            key: (*k).clone(),
            msg: VhcMessage::Join(Join {
                run_id: "withhold".into(),
                iroh_id: IrohId([0x44; 32]),
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
                envelope_hash: None,
            }),
        });
    }
    for k in [&w0, &w1] {
        script.push(ScriptMsg {
            key: (*k).clone(),
            msg: VhcMessage::Heartbeat(Heartbeat {
                round: 0,
                ready: Some(true),
            }),
        });
    }
    for round in 0..2u64 {
        let bytes = format!("update/withhold/0/{round}").into_bytes();
        let hash = blake3_hash(&bytes);
        payloads.insert(hash, bytes.clone());
        script.push(ScriptMsg {
            key: w0.clone(),
            msg: VhcMessage::Commitment(Commitment {
                round,
                payload: hash,
                size: bytes.len() as u64,
                locators: Vec::new(),
            }),
        });
        script.push(ScriptMsg {
            key: w0.clone(),
            msg: VhcMessage::StorageReceipt(StorageReceipt {
                round,
                verified: vec![RecordEntry {
                    peer: peer_id(&w0),
                    hash,
                    size: bytes.len() as u64,
                }],
            }),
        });
        // Fillers: the event-count clock crosses the round deadline deterministically.
        for _ in 0..8 {
            script.push(ScriptMsg {
                key: w0.clone(),
                msg: VhcMessage::Heartbeat(Heartbeat { round, ready: None }),
            });
        }
    }

    // The blob IS the oracle (consensus runs only in the sandboxed module): drive it under the
    // withholding and assert the SUBSTANCE on its OWN published records — both rounds close by the
    // deterministic event-count deadline WITHOUT the withheld peer, and it is dropped at k.
    let (_, _, decisions) = drive_and_archive(&wasm, &spec, key_seed, &custody, &script, 2);
    let records: Vec<&VhcMessage> = decisions
        .iter()
        .filter(|m| matches!(m, VhcMessage::RoundRecord(_)))
        .collect();
    assert_eq!(records.len(), 2, "both withheld rounds still close");
    for r in &records {
        let VhcMessage::RoundRecord(rr) = r else {
            unreachable!()
        };
        let entries = rr.inline.clone().unwrap_or_default();
        assert_eq!(entries.len(), 1, "the record excludes the withheld peer");
        assert_ne!(entries[0].peer, withheld_peer);
    }
    let VhcMessage::RoundRecord(last) = records[1] else {
        unreachable!()
    };
    assert_eq!(
        last.drops,
        vec![withheld_peer],
        "the withheld peer is dropped at k_absences = 2"
    );
}

// -- lane 3: partition (eclipse) heal — the eclipsed peer converges from the archive alone ---------

#[test]
fn eclipsed_peer_converges_from_archive_and_payloads_alone() {
    let wasm = coordinator_quorum_wasm();
    let custody = SigningKey::from_bytes(blake3::hash(b"eclipse/custody").as_bytes());
    let authority = single_key(peer_id(&custody));
    let run_id = Hash(*blake3::hash(b"eclipse/run").as_bytes());
    let worker_keys = [
        SigningKey::from_bytes(blake3::hash(b"eclipse/w0").as_bytes()),
        SigningKey::from_bytes(blake3::hash(b"eclipse/w1").as_bytes()),
    ];
    let initial =
        CoordinatorState::new(run_config("eclipse", 2, 1_000_000, 8), Seed([0x33; 32]), 0);
    let spec = spec_for(&wasm, &initial, authority, run_id);
    let key_seed = *blake3::hash(b"eclipse/frame-key").as_bytes();

    let mut payloads = BTreeMap::new();
    let script = barrier_script("eclipse", &worker_keys, 0..3, "E", &mut payloads);
    let (archive, heads, decisions) =
        drive_and_archive(&wasm, &spec, key_seed, &custody, &script, 3);

    // The partitioned peer saw NOTHING live. When the partition heals it holds only the archive
    // replica + payload store — and re-derives the ENTIRE history, digests included.
    let sandbox = SandboxedCoordinator::new(wasm.clone());
    let report = replay_consensus_from_archive(&sandbox, &archive, &heads, &payloads)
        .expect("heal replay green");
    assert_eq!(report.replay.rounds_verified, 3);
    assert_eq!(report.set_commitments_verified, 3);

    // Convergence: the re-derived decision stream equals the live run's exact decision stream. The
    // resync anchor is a blake3 of the coordinator's published `RoundRecord`s (the observer sees
    // published objects, never the module's privileged internal state), so the reference is that
    // same anchor over the LIVE blob's own published records — no native oracle.
    let final_ref = {
        let records: Vec<_> = decisions
            .iter()
            .filter_map(|m| match m {
                VhcMessage::RoundRecord(r) => Some(r.clone()),
                _ => None,
            })
            .collect();
        blake3_hash(&to_canonical_vec(&records).unwrap())
    };
    assert_eq!(
        report.replay.final_state_hash, final_ref,
        "the eclipsed peer converges to the live coordinator's exact decision stream"
    );
    assert_eq!(
        decisions
            .iter()
            .filter(|m| matches!(m, VhcMessage::RoundRecord(_)))
            .count(),
        3
    );
}

// -- lane 4: ThresholdKeys — quorum refusals end-to-end (the m-of-n lane) --------------------------

#[test]
#[allow(clippy::too_many_lines)]
fn threshold_keys_quorum_refusals_end_to_end() {
    let wasm = coordinator_quorum_wasm();
    // A 2-of-3 custody set holds head/record authority; the blob still runs the round logic.
    let m1 = SigningKey::from_bytes(blake3::hash(b"tk/member-1").as_bytes());
    let m2 = SigningKey::from_bytes(blake3::hash(b"tk/member-2").as_bytes());
    let m3 = SigningKey::from_bytes(blake3::hash(b"tk/member-3").as_bytes());
    let outsider = SigningKey::from_bytes(blake3::hash(b"tk/outsider").as_bytes());
    let authority = AuthorityConfig {
        topology: Topology::ThresholdKeys(
            ThresholdKeys::new(vec![peer_id(&m1), peer_id(&m2), peer_id(&m3)], 2).unwrap(),
        ),
        records_channel: DEFAULT_RECORDS_CHANNEL,
    };
    let run_id = Hash(*blake3::hash(b"tk/run").as_bytes());
    let worker_keys = [
        SigningKey::from_bytes(blake3::hash(b"tk/w0").as_bytes()),
        SigningKey::from_bytes(blake3::hash(b"tk/w1").as_bytes()),
    ];
    let initial = CoordinatorState::new(run_config("tk", 2, 1_000_000, 8), Seed([0x33; 32]), 0);
    let spec = spec_for(&wasm, &initial, authority.clone(), run_id);
    let key_seed = *blake3::hash(b"tk/frame-key").as_bytes();

    // Drive the blob; the single-custody heads drive_and_archive attests are correctly
    // SUB-QUORUM under the threshold topology (asserted below); the quorum re-attests them.
    let mut payloads = BTreeMap::new();
    let script = barrier_script("tk", &worker_keys, 0..2, "T", &mut payloads);
    let (source_archive, single_heads, _) =
        drive_and_archive(&wasm, &spec, key_seed, &m1, &script, 2);

    // A threshold archive replica: same segments, heads re-attested by the 2-of-3 quorum.
    let mut archive = RecordArchive::new(
        authority.clone(),
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    let mut quorum_heads = Vec::new();
    for h in &single_heads {
        let bytes = source_archive
            .fetch(&h.body.segment_hash)
            .expect("segment")
            .to_vec();
        archive.publish_segment(bytes).expect("publish");
        let head = AttestedHead::attest(&[m1.clone(), m2.clone()], h.body.clone()).unwrap();
        archive
            .ingest_head(head.clone())
            .expect("quorum head accepted");
        quorum_heads.push(head);
    }

    // END-TO-END positive: consensus replay green under the threshold topology.
    let sandbox = SandboxedCoordinator::new(wasm.clone());
    let report = replay_consensus_from_archive(&sandbox, &archive, &quorum_heads, &payloads)
        .expect("threshold replay green");
    assert_eq!(report.replay.rounds_verified, 2);

    // The typed quorum refusals (D1's AuthError, asserted per guard):
    let body = quorum_heads[0].body.clone();
    let preimage = quorum_heads[0].preimage().unwrap();
    // (a) sub-quorum: one member signature < m=2 — including drive_and_archive's single-custody
    // head, which must NOT pass the threshold archive.
    let one_sig = AttestedHead::single(&m1, body.clone()).unwrap();
    assert_eq!(
        authority.authorize(&preimage, &one_sig.sigs),
        Err(AuthError::InsufficientSignatures { have: 1, need: 2 })
    );
    let mut fresh_archive = RecordArchive::new(
        authority.clone(),
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    assert!(matches!(
        fresh_archive.ingest_head(one_sig).unwrap_err(),
        ArchiveError::Unauthoritative(_)
    ));
    // (b) duplicate-signer inflation: the same member twice cannot reach the quorum.
    let dup = AttestedHead::attest(&[m1.clone(), m1.clone()], body.clone()).unwrap();
    assert_eq!(
        authority.authorize(&preimage, &dup.sigs),
        Err(AuthError::DuplicateSigner {
            signer: peer_id(&m1)
        })
    );
    // (c) a non-member cannot contribute.
    let with_outsider =
        AttestedHead::attest(&[m1.clone(), outsider.clone()], body.clone()).unwrap();
    assert_eq!(
        authority.authorize(&preimage, &with_outsider.sigs),
        Err(AuthError::UnknownSigner {
            signer: peer_id(&outsider)
        })
    );
    // (d) a bad signature from a genuine member is typed, not counted.
    let mut forged = AttestedHead::attest(&[m1.clone(), m2.clone()], body.clone()).unwrap();
    forged.sigs[1].sig.0[0] ^= 0xff;
    assert_eq!(
        authority.authorize(&preimage, &forged.sigs),
        Err(AuthError::BadSignature {
            signer: peer_id(&m2)
        })
    );
    // (e) the layering pin: a single §12.1 TRANSPORT signature can never satisfy a threshold
    // topology — record authority rides record-level signature sets, not transport frames.
    let sig = sign_canonical(&m1, &42u64).unwrap();
    let single_transport = [RecordSig {
        signer: peer_id(&m1),
        sig,
    }];
    assert!(matches!(
        authority.authorize(&to_canonical_vec(&42u64).unwrap(), &single_transport),
        Err(AuthError::InsufficientSignatures { .. })
    ));
    // (f) equivocation under threshold: two QUORUM-attested divergent heads still meet as
    // portable evidence — quorum intersection (2m > n) pins at least one member on both sides.
    let mut divergent = body;
    divergent.segment_hash = Hash([0xEE; 32]);
    let head_x = quorum_heads[0].clone();
    let head_y = AttestedHead::attest(&[m2.clone(), m3.clone()], divergent).unwrap();
    let evidence = detect_fork(&[head_x, head_y], &authority).expect("threshold fork evidence");
    assert!(matches!(evidence, ForkEvidence::DivergentHead { .. }));
}

// -- lane 5: decay-while-waiting through the blob (the C2 floor-breach wedge, reliability §7) ------

/// The C2 floor-breach membership wedge, healed by the production `coordinator_quorum.wasm`
/// blob itself (the native repair is pinned in `decay_while_waiting.rs`; this lane proves the
/// SAME tick ships in the sandboxed module the fleet actually runs). Scripted entirely on the
/// deterministic event-count clock with tiny authored deadlines:
///
/// 1. a, b, c train round 0; b and c go silent and drop by ROUND accounting at k_absences = 2
///    (round 1's record carries the drops); the floor breach (1 < 3) parks the run
///    `Cooldown → WaitingForMembers`.
/// 2. d joins during the wait and never speaks again — the zombie seat that, pre-repair, NO
///    mechanism could reclaim: absence accounting lived only in round finalization and the
///    waiting phase has no rounds. The wait stretches past the round-scaled decay window
///    (k × (train + witness) = 10 ticks) while a keeps heartbeating; d decays WITHOUT a round
///    (a keeps its seat — liveness re-stamps).
/// 3. e and f join through the ordinary path, the floor gathers (a+e+f), warmup exits on
///    readiness, and **round 2 opens and closes with exactly the healed roster** — the
///    observable that discriminates the repair: pre-repair, zombie d stays "healthy", warmup
///    can never gather full readiness, and no round ever opens again.
#[test]
fn floor_breach_zombie_decays_while_waiting_and_the_healed_roster_trains() {
    let wasm = coordinator_quorum_wasm();
    let custody = SigningKey::from_bytes(blake3::hash(b"decay/custody").as_bytes());
    let authority = single_key(peer_id(&custody));
    let run_id = Hash(*blake3::hash(b"decay/run").as_bytes());
    let key = |label: &str| SigningKey::from_bytes(blake3::hash(label.as_bytes()).as_bytes());
    let (a, b, c, d, e, f) = (
        key("decay/a"),
        key("decay/b"),
        key("decay/c"),
        key("decay/d"),
        key("decay/e"),
        key("decay/f"),
    );

    // Tiny authored deadlines (NOT the harness's effectively-infinite event_clock defaults):
    // train 3 + witness 2 → decay window k(2) × 5 = 10 ticks; cooldown 2 so the breach can
    // reach WaitingForMembers on the event-count clock.
    let mut cfg = run_config("decay", 3, 3, 2);
    cfg.round_witness_s = 2;
    cfg.cooldown_s = 2;
    let initial = CoordinatorState::new(cfg, Seed([0x33; 32]), 0);
    let spec = spec_for(&wasm, &initial, authority, run_id);
    let key_seed = *blake3::hash(b"decay/frame-key").as_bytes();

    let join = |k: &SigningKey| ScriptMsg {
        key: k.clone(),
        msg: VhcMessage::Join(Join {
            run_id: "decay".into(),
            iroh_id: IrohId([0x44; 32]),
            class: ThroughputClass::C1,
            capabilities: CapabilitySet::new(),
            envelope_hash: None,
        }),
    };
    let ready = |k: &SigningKey, round: u64| ScriptMsg {
        key: k.clone(),
        msg: VhcMessage::Heartbeat(Heartbeat {
            round,
            ready: Some(true),
        }),
    };
    let filler = |k: &SigningKey, round: u64| ScriptMsg {
        key: k.clone(),
        msg: VhcMessage::Heartbeat(Heartbeat { round, ready: None }),
    };

    let mut payloads = BTreeMap::new();
    let mut script = Vec::new();
    // Prologue: a, b, c join + ready → round 0 opens on the readiness fast path.
    for k in [&a, &b, &c] {
        script.push(join(k));
    }
    for k in [&a, &b, &c] {
        script.push(ready(k, 0));
    }
    // Rounds 0 and 1: only a commits (b and c are silent); each round closes at the
    // event-count train deadline and finalizes off the evidenced committed set.
    for round in 0..2u64 {
        let bytes = format!("update/decay/a/{round}").into_bytes();
        let hash = blake3_hash(&bytes);
        payloads.insert(hash, bytes.clone());
        script.push(ScriptMsg {
            key: a.clone(),
            msg: VhcMessage::Commitment(Commitment {
                round,
                payload: hash,
                size: bytes.len() as u64,
                locators: Vec::new(),
            }),
        });
        script.push(ScriptMsg {
            key: a.clone(),
            msg: VhcMessage::StorageReceipt(StorageReceipt {
                round,
                verified: vec![RecordEntry {
                    peer: peer_id(&a),
                    hash,
                    size: bytes.len() as u64,
                }],
            }),
        });
        for _ in 0..4 {
            script.push(filler(&a, round));
        }
    }
    // The breach cools down (2 ticks) into WaitingForMembers.
    for _ in 0..4 {
        script.push(filler(&a, 1));
    }
    // d joins during the wait, then NEVER speaks again — the zombie seat.
    script.push(join(&d));
    // The wait stretches past the decay window (10 ticks from d's join) on a's heartbeats:
    // a re-stamps its own liveness every filler, d decays without a round.
    for _ in 0..14 {
        script.push(filler(&a, 1));
    }
    // The freed floor gathers through the ordinary Join path and warms up on readiness.
    script.push(join(&e));
    script.push(join(&f));
    for k in [&a, &e, &f] {
        script.push(ready(k, 2));
    }
    // Round 2 trains on the HEALED roster: all three commit, one receipt evidences the set —
    // the all-committed + all-evidenced fast paths finalize with no fillers.
    let mut entries = Vec::new();
    for (i, k) in [&a, &e, &f].into_iter().enumerate() {
        let bytes = format!("update/decay/healed-{i}/2").into_bytes();
        let hash = blake3_hash(&bytes);
        payloads.insert(hash, bytes.clone());
        entries.push(RecordEntry {
            peer: peer_id(k),
            hash,
            size: bytes.len() as u64,
        });
        script.push(ScriptMsg {
            key: (*k).clone(),
            msg: VhcMessage::Commitment(Commitment {
                round: 2,
                payload: hash,
                size: bytes.len() as u64,
                locators: Vec::new(),
            }),
        });
    }
    script.push(ScriptMsg {
        key: a.clone(),
        msg: VhcMessage::StorageReceipt(StorageReceipt {
            round: 2,
            verified: entries,
        }),
    });

    // The blob IS the oracle. Nine records drain deterministically: the two breach rounds, the
    // catch-up REPLAY-FORWARD (each join during the wait re-publishes records 0 and 1 for the
    // rejoiner — d, e, f × 2), and the healed round 2. The wedged shape never publishes a
    // round-2 record at all (zombie d blocks warmup readiness forever).
    let (_, _, decisions) = drive_and_archive(&wasm, &spec, key_seed, &custody, &script, 9);
    let records: Vec<_> = decisions
        .iter()
        .filter_map(|m| match m {
            VhcMessage::RoundRecord(rr) => Some(rr),
            _ => None,
        })
        .collect();
    // Round 0: a's evidence made the record (every re-publish is byte-identical in substance).
    for rr in records.iter().filter(|rr| rr.round == 0) {
        let entries = rr.inline.clone().unwrap_or_default();
        assert_eq!(entries.len(), 1, "round 0 carries only a's entry");
        assert_eq!(entries[0].peer, peer_id(&a));
    }
    // Round 1 closes EMPTY by the deadline (a's commit landed late under the tightened window —
    // irrelevant to the subject) and drops the silent trainers at k_absences = 2.
    let r1 = records
        .iter()
        .find(|rr| rr.round == 1)
        .expect("round 1 record");
    assert!(r1.inline.clone().unwrap_or_default().is_empty());
    let mut dropped = r1.drops.clone();
    dropped.sort();
    let mut expected = vec![peer_id(&b), peer_id(&c)];
    expected.sort();
    assert_eq!(
        dropped, expected,
        "round accounting drops the dead trainers"
    );
    // Round 2 — the discriminator: it exists at all (zombie d decayed while waiting; otherwise
    // warmup never gathers readiness), and it closes with exactly the healed roster.
    let r2: Vec<_> = records.iter().filter(|rr| rr.round == 2).collect();
    assert_eq!(r2.len(), 1, "one healed round record");
    let mut healed: Vec<PeerId> = r2[0]
        .inline
        .clone()
        .unwrap_or_default()
        .iter()
        .map(|e| e.peer)
        .collect();
    healed.sort();
    let mut expected_roster = vec![peer_id(&a), peer_id(&e), peer_id(&f)];
    expected_roster.sort();
    assert_eq!(
        healed, expected_roster,
        "round 2 trains a + e + f, zombie-free"
    );
    assert!(r2[0].drops.is_empty(), "the healed round drops nobody");
}
