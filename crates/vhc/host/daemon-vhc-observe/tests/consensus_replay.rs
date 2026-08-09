// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **Consensus replay — the third replay tier** (architecture §3.6; refactor §8/D2, gated in
// tier-2): a third party holding ONLY the record archive (signed, hash-chained, content-addressed
// sealed segments) and the content-addressed payloads re-verifies every consensus decision and
// every digest. No live journal, no coordinator process, no privileged input.
//
// The fixture drives a real multi-round run through the sandboxed `coordinator-quorum` module
// (consensus never runs natively, even to build the fixture), journaling every input (tag 1), the
// initial state (tag 10), and every signed publish (tag 4) into the on-disk A1 journal with a small
// rotation threshold (multiple sealed segments), then publishes the sealed chain into a
// `RecordArchive` under signed heads — and verifies from there alone, re-deriving inside the sandbox.
//
// Sanctioned raw-fs test home (the journal writes real files): the journal-tests pattern.
#![allow(clippy::disallowed_methods)]

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
use daemon_vhc_proto::{peer_id, to_canonical_vec, Hash, PeerId, VHC_PROTO_VERSION};
use daemon_vhc_sdk_consensus::coordinator::Input;
use daemon_vhc_sdk_consensus::SignedMessage;
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};

mod common;
use common::{coord_key, coordinator_sandbox, run_fixture, Fixture};

const ROUNDS: u64 = 6;
const RUN_LABEL: &str = "consensus-replay";

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

/// The archived-run fixture: drive the coordinator module over a ROUNDS-round script, journaling
/// inputs + publishes; seal the chain; publish it to an archive under signed heads.
struct Archived {
    fx: Fixture,
    archive: RecordArchive,
    heads: Vec<AttestedHead>,
    authority: AuthorityConfig,
}

fn single_key(authority: PeerId) -> AuthorityConfig {
    AuthorityConfig {
        topology: Topology::SingleKey(SingleKey::new(authority)),
        records_channel: DEFAULT_RECORDS_CHANNEL,
    }
}

fn build_archived(fx: Fixture) -> Archived {
    let coord = coord_key();
    let authority = single_key(peer_id(&coord));

    let root = tempdir();
    let mut journal = Journal::create(
        &root,
        ident(),
        StaticKey::new([7u8; 32]),
        // Small threshold so the run spans several sealed segments (the chain the archive signs).
        RotatePolicy {
            max_records: 8,
            ..RotatePolicy::default()
        },
    )
    .expect("journal create");

    record_run_header(&mut journal, &ident(), Vec::new()).expect("run header");
    record_initial_state(&mut journal, &fx.initial).expect("snapshot");

    // Journal the driving inputs (tag 1), then the module's published decisions (tag 4, signed by
    // the coordinator authority as it broadcasts them). The module owns its clock, so no clock
    // records are journaled — the sandbox re-derives them on replay.
    for (at, sm) in fx.driving.iter().enumerate() {
        record_input(&mut journal, at as u64, &Input::Message(sm.clone())).expect("record input");
    }
    for msg in &fx.published {
        let payload = to_canonical_vec(msg).expect("payload cbor");
        let signed =
            SignedMessage::sign(&coord, VHC_PROTO_VERSION, msg.clone()).expect("sign publish");
        let frame = to_canonical_vec(&signed).expect("frame cbor");
        journal
            .publish(0, &payload, frame)
            .expect("journal publish");
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
            &coord,
            ChainHead {
                run_id: ident().run_id,
                epoch: 0,
                role: "coordinator".into(),
                instance: 0,
                module: ident().module,
                segment: ord,
                segment_hash: addr,
                prev_hash: prev,
                // The production head convention (§8.8): the record count EXCLUDES the seal —
                // the head↔segment binding refuses a head that counts the seal record.
                records: scan.records.len() as u64 - 1,
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

    Archived {
        fx,
        archive,
        heads,
        authority,
    }
}

fn build_fixture(sandbox: &common::SandboxedCoordinator) -> Archived {
    let fx = run_fixture(sandbox, RUN_LABEL, ROUNDS);
    build_archived(fx)
}

#[test]
fn third_party_reverifies_digests_from_archive_and_payloads_alone() {
    let sandbox = coordinator_sandbox();
    let ar = build_fixture(&sandbox);
    let report = replay_consensus_from_archive(&sandbox, &ar.archive, &ar.heads, &ar.fx.payloads)
        .expect("consensus replay green");

    assert_eq!(report.segments_verified as usize, ar.heads.len());
    // Every archived RoundRecord re-derived byte-identically inside the sandbox (the oracle), and
    // every round's digest recomputed from the payloads alone.
    assert_eq!(report.replay.rounds_verified, ROUNDS);
    assert_eq!(report.set_commitments_verified, ROUNDS);
    assert_eq!(report.payload_entries_verified, ROUNDS * 2);
}

#[test]
fn missing_payload_is_typed_incomplete_never_a_pass() {
    let sandbox = coordinator_sandbox();
    let ar = build_fixture(&sandbox);
    let mut incomplete = ar.fx.payloads.clone();
    let (&victim, _) = incomplete.iter().next().expect("payloads");
    incomplete.remove(&victim);

    let err =
        replay_consensus_from_archive(&sandbox, &ar.archive, &ar.heads, &incomplete).unwrap_err();
    assert!(
        matches!(err, ConsensusReplayError::MissingPayload { hash, .. } if hash == victim),
        "got {err}"
    );
}

#[test]
fn withheld_segment_breaks_the_walk_typed() {
    let sandbox = coordinator_sandbox();
    let ar = build_fixture(&sandbox);
    // A replica that never received the middle segment: fetch fails typed.
    let mut partial = RecordArchive::new(
        ar.authority.clone(),
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    for (i, head) in ar.heads.iter().enumerate() {
        if i == 1 {
            continue; // withhold segment 1's bytes
        }
        let bytes = ar
            .archive
            .fetch(&head.body.segment_hash)
            .expect("source has it")
            .to_vec();
        partial.publish_segment(bytes).expect("publish");
    }
    let err =
        replay_consensus_from_archive(&sandbox, &partial, &ar.heads, &ar.fx.payloads).unwrap_err();
    assert!(
        matches!(err, ConsensusReplayError::MissingSegment { segment: 1, .. }),
        "got {err}"
    );
}

#[test]
fn forged_or_gappy_heads_are_typed_refusals() {
    let sandbox = coordinator_sandbox();
    let ar = build_fixture(&sandbox);

    // A head attested by an impostor fails AuthorityConfig::authorize (WrongSigner under the
    // SingleKey topology — D1's typed refusal).
    let impostor = common::key(9);
    let mut forged = ar.heads.clone();
    forged[0] = AttestedHead::single(&impostor, forged[0].body.clone()).expect("forged head");
    let err =
        replay_consensus_from_archive(&sandbox, &ar.archive, &forged, &ar.fx.payloads).unwrap_err();
    assert!(
        matches!(err, ConsensusReplayError::Unauthoritative { segment: 0 }),
        "got {err}"
    );

    // Heads missing segment 0 cannot anchor the chain.
    let gappy: Vec<AttestedHead> = ar.heads[1..].to_vec();
    let err =
        replay_consensus_from_archive(&sandbox, &ar.archive, &gappy, &ar.fx.payloads).unwrap_err();
    assert!(
        matches!(err, ConsensusReplayError::ChainBroken { segment: 0, .. }),
        "got {err}"
    );
}
