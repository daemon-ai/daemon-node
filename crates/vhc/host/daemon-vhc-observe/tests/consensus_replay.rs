// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **Consensus replay — the third replay tier** (architecture §3.6; refactor §8/D2, gated in
// tier-2): a third party holding ONLY the record archive (signed, hash-chained, content-addressed
// sealed segments) and the content-addressed payloads re-verifies every consensus decision and
// every digest. No live journal, no coordinator, no privileged input.
//
// The fixture drives a real multi-round run through the pure coordinator `tick`, journaling every
// input (tag 1/3), the initial state (tag 10), and every signed publish (tag 4) into the on-disk
// A1 journal with a small rotation threshold (multiple sealed segments), then publishes the
// sealed chain into a `RecordArchive` under signed heads — and verifies from there alone.
//
// Sanctioned raw-fs test home (the journal writes real files): the journal-tests pattern.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;

use daemon_vhc_observe::journal::archive::{ChainHead, RecordArchive};
use daemon_vhc_observe::journal::oracle::{record_initial_state, record_input, record_run_header};
use daemon_vhc_observe::journal::record::ExecIdentity;
use daemon_vhc_observe::journal::segment::scan_file;
use daemon_vhc_observe::journal::store::{Journal, RotatePolicy};
use daemon_vhc_observe::journal::StaticKey;
use daemon_vhc_observe::{
    replay_consensus_from_archive, AttestedHead, ConsensusReplayError, ReplicationPolicy,
    RetentionPolicy,
};
use daemon_vhc_proto::envelope::{GlobalBatch, StopCondition};
use daemon_vhc_proto::messages::{
    Commitment, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass,
};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId, PeerId, Seed,
    SignedMessage, SigningKey, SwarmMessage, SWARM_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{tick, CoordinatorState, Input, Output, RunConfig};
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};

const ROUNDS: u64 = 6;
const RUN_LABEL: &str = "consensus-replay";

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn ident() -> ExecIdentity {
    ExecIdentity {
        run_id: Hash([0xAA; 32]),
        epoch: 0,
        role: "coordinator".into(),
        instance: 0,
        module: Hash([0xBB; 32]),
    }
}

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    base.push(format!(
        "dvhc-consensus-replay-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// The run's coordinator config: continuous rounds, huge deadlines (event-driven fast paths only).
fn run_config() -> RunConfig {
    RunConfig {
        run_id: RUN_LABEL.into(),
        proto_version: SWARM_PROTO_VERSION,
        envelope_hash: Hash([0u8; 32]),
        required_capabilities: CapabilitySet::new(),
        min_peers: 2,
        max_peers: 4,
        warmup_s: 1_000_000,
        round_train_max_s: 1_000_000,
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
        k_absences: 8,
        verification_percent: 0,
        authorized: Vec::new(),
    }
}

/// The archived-run fixture: drive `tick` over a ROUNDS-round script, journaling inputs +
/// publishes; seal the chain; publish it to an archive under signed heads. Returns the archive,
/// the heads, the payload table, and the coordinator authority.
struct Fixture {
    archive: RecordArchive,
    heads: Vec<AttestedHead>,
    payloads: BTreeMap<Hash, Vec<u8>>,
    authority: AuthorityConfig,
}

fn single_key(authority: PeerId) -> AuthorityConfig {
    AuthorityConfig {
        topology: Topology::SingleKey(SingleKey::new(authority)),
        records_channel: DEFAULT_RECORDS_CHANNEL,
    }
}

fn build_fixture() -> Fixture {
    let coord_key = key(1);
    let authority = single_key(peer_id(&coord_key));
    let worker_keys = [key(2), key(3)];
    let peers: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();

    let root = tempdir();
    let mut journal = Journal::create(
        &root,
        ident(),
        StaticKey::new([7u8; 32]),
        // Small threshold so the run spans several sealed segments (the chain the archive signs).
        RotatePolicy { max_records: 8 },
    )
    .expect("journal create");

    let initial = CoordinatorState::new(run_config(), Seed([0x33; 32]), 0);
    record_run_header(&mut journal, &ident(), Vec::new()).expect("run header");
    record_initial_state(&mut journal, &initial).expect("snapshot");

    // The drive: journal each input (tag 1/3), tick, journal each publish (tag 4, signed).
    let mut state = initial;
    let mut at = 0u64;
    let mut now_s = 0u64;
    let mut payloads: BTreeMap<Hash, Vec<u8>> = BTreeMap::new();
    let feed = |journal: &mut Journal<StaticKey>,
                state: &mut CoordinatorState,
                at: &mut u64,
                now_s: &mut u64,
                signed: SignedMessage| {
        for input in [Input::Message(signed), {
            *now_s += 1;
            Input::Clock(*now_s)
        }] {
            record_input(journal, *at, &input).expect("record input");
            *at += 1;
            let (next, outputs) = tick(state.clone(), input);
            *state = next;
            for out in outputs {
                if let Output::Publish(msg) = out {
                    let payload = to_canonical_vec(&*msg).expect("payload cbor");
                    let signed =
                        SignedMessage::sign(&coord_key, SWARM_PROTO_VERSION, (*msg).clone())
                            .expect("sign publish");
                    let frame = to_canonical_vec(&signed).expect("frame cbor");
                    journal
                        .publish(0, &payload, frame)
                        .expect("journal publish");
                }
            }
        }
    };

    let sign = |k: &SigningKey, m: SwarmMessage| {
        SignedMessage::sign(k, SWARM_PROTO_VERSION, m).expect("sign")
    };

    // Joins + ready heartbeats.
    for k in &worker_keys {
        feed(
            &mut journal,
            &mut state,
            &mut at,
            &mut now_s,
            sign(
                k,
                SwarmMessage::Join(Join {
                    run_id: RUN_LABEL.into(),
                    iroh_id: IrohId([0x44; 32]),
                    class: ThroughputClass::C1,
                    capabilities: CapabilitySet::new(),
                    envelope_hash: None,
                }),
            ),
        );
    }
    for k in &worker_keys {
        feed(
            &mut journal,
            &mut state,
            &mut at,
            &mut now_s,
            sign(
                k,
                SwarmMessage::Heartbeat(Heartbeat {
                    round: 0,
                    ready: Some(true),
                }),
            ),
        );
    }

    // Rounds: two commitments (with REAL payload bytes) + one covering receipt per round.
    for round in 0..ROUNDS {
        let mut entries = Vec::new();
        for (i, k) in worker_keys.iter().enumerate() {
            let bytes = format!("update/{i}/{round}").into_bytes();
            let hash = blake3_hash(&bytes);
            payloads.insert(hash, bytes.clone());
            feed(
                &mut journal,
                &mut state,
                &mut at,
                &mut now_s,
                sign(
                    k,
                    SwarmMessage::Commitment(Commitment {
                        round,
                        payload: hash,
                        size: bytes.len() as u64,
                        locators: Vec::new(),
                    }),
                ),
            );
            entries.push(RecordEntry {
                peer: peers[i],
                hash,
                size: bytes.len() as u64,
            });
        }
        feed(
            &mut journal,
            &mut state,
            &mut at,
            &mut now_s,
            sign(
                &worker_keys[0],
                SwarmMessage::StorageReceipt(StorageReceipt {
                    round,
                    verified: entries,
                }),
            ),
        );
    }

    // Seal the tail so the WHOLE run is archived (the archive holds sealed segments only).
    journal.roll().expect("final roll");

    // Publish every sealed segment content-addressed + build the attested head chain.
    let mut archive = RecordArchive::new(
        authority.clone(),
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    let mut heads = Vec::new();
    let mut prev = Hash([0u8; 32]);
    let ords = journal.paths().existing_segments().expect("segments");
    for &ord in &ords {
        let scan = scan_file(journal.paths().segment(ord)).expect("scan");
        if !scan.sealed {
            continue; // the freshly-rolled empty tail
        }
        let bytes = std::fs::read(journal.paths().segment(ord)).expect("read segment");
        let addr = archive.publish_segment(bytes).expect("publish");
        let head = AttestedHead::single(
            &coord_key,
            ChainHead {
                run_id: ident().run_id,
                epoch: 0,
                role: "coordinator".into(),
                instance: 0,
                module: ident().module,
                segment: ord,
                segment_hash: addr,
                prev_hash: prev,
                records: scan.records.len() as u64,
            },
        )
        .expect("attest head");
        archive.ingest_head(head.clone()).expect("head accepted");
        heads.push(head);
        prev = addr;
    }
    assert!(
        heads.len() >= 3,
        "the fixture must span several sealed segments (got {})",
        heads.len()
    );

    Fixture {
        archive,
        heads,
        payloads,
        authority,
    }
}

#[test]
fn third_party_reverifies_digests_from_archive_and_payloads_alone() {
    let fx = build_fixture();
    let report = replay_consensus_from_archive(&fx.archive, &fx.heads, &fx.payloads)
        .expect("consensus replay green");

    assert_eq!(report.segments_verified as usize, fx.heads.len());
    // Every archived RoundRecord re-derived byte-identically (the oracle), and every round's
    // digest recomputed from the payloads alone.
    assert_eq!(report.replay.rounds_verified, ROUNDS);
    assert_eq!(report.set_commitments_verified, ROUNDS);
    assert_eq!(report.payload_entries_verified, ROUNDS * 2);
}

#[test]
fn missing_payload_is_typed_incomplete_never_a_pass() {
    let fx = build_fixture();
    let mut incomplete = fx.payloads.clone();
    let (&victim, _) = incomplete.iter().next().expect("payloads");
    incomplete.remove(&victim);

    let err = replay_consensus_from_archive(&fx.archive, &fx.heads, &incomplete).unwrap_err();
    assert!(
        matches!(err, ConsensusReplayError::MissingPayload { hash, .. } if hash == victim),
        "got {err}"
    );
}

#[test]
fn withheld_segment_breaks_the_walk_typed() {
    let fx = build_fixture();
    // A replica that never received the middle segment: fetch fails typed.
    let mut partial = RecordArchive::new(
        fx.authority,
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    for (i, head) in fx.heads.iter().enumerate() {
        if i == 1 {
            continue; // withhold segment 1's bytes
        }
        let bytes = fx
            .archive
            .fetch(&head.body.segment_hash)
            .expect("source has it")
            .to_vec();
        partial.publish_segment(bytes).expect("publish");
    }
    let err = replay_consensus_from_archive(&partial, &fx.heads, &fx.payloads).unwrap_err();
    assert!(
        matches!(err, ConsensusReplayError::MissingSegment { segment: 1, .. }),
        "got {err}"
    );
}

#[test]
fn forged_or_gappy_heads_are_typed_refusals() {
    let fx = build_fixture();

    // A head attested by an impostor fails AuthorityConfig::authorize (WrongSigner under the
    // SingleKey topology — D1's typed refusal).
    let impostor = key(9);
    let mut forged = fx.heads.clone();
    forged[0] = AttestedHead::single(&impostor, forged[0].body.clone()).expect("forged head");
    let err = replay_consensus_from_archive(&fx.archive, &forged, &fx.payloads).unwrap_err();
    assert!(
        matches!(err, ConsensusReplayError::Unauthoritative { segment: 0 }),
        "got {err}"
    );

    // Heads missing segment 0 cannot anchor the chain.
    let gappy: Vec<AttestedHead> = fx.heads[1..].to_vec();
    let err = replay_consensus_from_archive(&fx.archive, &gappy, &fx.payloads).unwrap_err();
    assert!(
        matches!(err, ConsensusReplayError::ChainBroken { segment: 0, .. }),
        "got {err}"
    );
}
