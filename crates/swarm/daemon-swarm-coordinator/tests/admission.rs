// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Admission: capability subset, version, envelope hash, roster capacity (spec §6.5; TDD PROTO-12/13).

mod common;

use common::*;
use daemon_swarm_proto::messages::{Join, ThroughputClass};
use daemon_swarm_proto::{
    peer_id, CapabilitySet, Hash, IrohId, SwarmProtoVersion, SWARM_PROTO_VERSION,
};

use daemon_swarm_coordinator::admission::{admit, JoinCandidate};
use daemon_swarm_coordinator::{AdmissionReject, Phase, RunConfig};

fn caps(tokens: &[&str]) -> CapabilitySet {
    CapabilitySet::from_tokens(tokens.iter().copied()).unwrap()
}

fn join_with(caps: CapabilitySet) -> Join {
    Join {
        run_id: RUN_ID.to_string(),
        iroh_id: IrohId([9; 32]),
        class: ThroughputClass::C3,
        capabilities: caps,
    }
}

fn cfg_requiring(tokens: &[&str]) -> RunConfig {
    let mut c = base_config();
    c.required_capabilities = caps(tokens);
    c
}

#[test]
fn proto12_capability_subset_admits() {
    let cfg = cfg_requiring(&["tensor-abi@1", "adamw_step@1"]);
    let j = join_with(caps(&["tensor-abi@1", "adamw_step@1", "flash_attn@1"]));
    let cand = JoinCandidate {
        peer: pid(1),
        version: SWARM_PROTO_VERSION,
        join: &j,
        asserted_hash: None,
    };
    assert!(admit(&cfg, Phase::WaitingForMembers, &[], &[], &cand).is_ok());
}

#[test]
fn proto12_missing_capability_rejected() {
    let cfg = cfg_requiring(&["tensor-abi@1", "adamw_step@1"]);
    let j = join_with(caps(&["tensor-abi@1"]));
    let cand = JoinCandidate {
        peer: pid(1),
        version: SWARM_PROTO_VERSION,
        join: &j,
        asserted_hash: None,
    };
    match admit(&cfg, Phase::WaitingForMembers, &[], &[], &cand) {
        Err(AdmissionReject::MissingCapabilities(missing)) => {
            assert_eq!(missing.len(), 1);
            assert_eq!(missing[0].token(), "adamw_step@1");
        }
        other => panic!("expected MissingCapabilities, got {other:?}"),
    }
}

#[test]
fn proto13_version_mismatch_rejected() {
    let cfg = base_config();
    let j = join_with(CapabilitySet::new());
    let cand = JoinCandidate {
        peer: pid(1),
        version: SwarmProtoVersion(999),
        join: &j,
        asserted_hash: None,
    };
    assert!(matches!(
        admit(&cfg, Phase::WaitingForMembers, &[], &[], &cand),
        Err(AdmissionReject::VersionMismatch { .. })
    ));
}

#[test]
fn envelope_hash_mismatch_rejected_when_asserted() {
    let cfg = base_config();
    let j = join_with(CapabilitySet::new());
    let wrong = Hash([0xEE; 32]);
    let cand = JoinCandidate {
        peer: pid(1),
        version: SWARM_PROTO_VERSION,
        join: &j,
        asserted_hash: Some(&wrong),
    };
    assert!(matches!(
        admit(&cfg, Phase::WaitingForMembers, &[], &[], &cand),
        Err(AdmissionReject::EnvelopeHashMismatch)
    ));

    // The run's real hash is admitted.
    let ok = cfg.envelope_hash;
    let cand = JoinCandidate {
        asserted_hash: Some(&ok),
        ..cand
    };
    assert!(admit(&cfg, Phase::WaitingForMembers, &[], &[], &cand).is_ok());
}

#[test]
fn envelope_hash_absent_is_tolerated() {
    // Back-compat: a join that asserts no envelope hash is admitted even though the run has a
    // specific frozen envelope hash. This is the wire reality until the `Join.envelope_hash` carrier
    // lands (a Merge-3-coordinated additive field — see swarm-ledger-p3.md); the enforcement above is
    // ready to consume it, but `tick` passes `None` today so existing peers are never spuriously
    // rejected.
    let cfg = base_config();
    assert_eq!(
        cfg.envelope_hash,
        Hash([0x11; 32]),
        "run has a real envelope hash"
    );
    let j = join_with(CapabilitySet::new());
    let cand = JoinCandidate {
        peer: pid(1),
        version: SWARM_PROTO_VERSION,
        join: &j,
        asserted_hash: None,
    };
    assert!(admit(&cfg, Phase::WaitingForMembers, &[], &[], &cand).is_ok());
}

#[test]
fn join_via_tick_does_not_yet_assert_envelope_hash() {
    // The `tick` join path threads `asserted_hash: None` (the frozen `Join` carries no hash yet), so
    // a join is admitted regardless of the run's envelope hash. When the additive `Join.envelope_hash`
    // field lands at Merge 3, `tick` will forward it and `EnvelopeHashMismatch` becomes reachable
    // from the wire (the reason + `admit` check are already present + tested above).
    let mut cfg = base_config();
    cfg.envelope_hash = Hash([0x77; 32]);
    let state = new_state(cfg);
    let (state, _) = daemon_swarm_coordinator::tick(
        state,
        daemon_swarm_coordinator::Input::Message(join_msg(&key(1))),
    );
    assert!(state.is_healthy_member(&peer_id(&key(1))));
}

#[test]
fn run_id_mismatch_rejected() {
    let cfg = base_config();
    let j = Join {
        run_id: "other-run".to_string(),
        iroh_id: IrohId([9; 32]),
        class: ThroughputClass::C2,
        capabilities: CapabilitySet::new(),
    };
    let cand = JoinCandidate {
        peer: pid(1),
        version: SWARM_PROTO_VERSION,
        join: &j,
        asserted_hash: None,
    };
    assert!(matches!(
        admit(&cfg, Phase::WaitingForMembers, &[], &[], &cand),
        Err(AdmissionReject::RunIdMismatch)
    ));
}

#[test]
fn roster_full_rejected() {
    let mut cfg = base_config();
    cfg.max_peers = 1;
    let existing =
        daemon_swarm_coordinator::Member::joining(pid(1), IrohId([1; 32]), ThroughputClass::C2, 0);
    let j = join_with(CapabilitySet::new());
    let cand = JoinCandidate {
        peer: pid(2),
        version: SWARM_PROTO_VERSION,
        join: &j,
        asserted_hash: None,
    };
    assert!(matches!(
        admit(&cfg, Phase::WaitingForMembers, &[existing], &[], &cand),
        Err(AdmissionReject::RosterFull)
    ));
}

#[test]
fn duplicate_healthy_peer_rejected_but_dropped_may_rejoin() {
    let cfg = base_config();
    let j = join_with(CapabilitySet::new());
    let healthy =
        daemon_swarm_coordinator::Member::joining(pid(1), IrohId([1; 32]), ThroughputClass::C2, 0);
    let cand = JoinCandidate {
        peer: pid(1),
        version: SWARM_PROTO_VERSION,
        join: &j,
        asserted_hash: None,
    };
    assert!(matches!(
        admit(
            &cfg,
            Phase::WaitingForMembers,
            std::slice::from_ref(&healthy),
            &[],
            &cand
        ),
        Err(AdmissionReject::DuplicatePeer)
    ));

    // A dropped member of the same identity may rejoin.
    let mut dropped = healthy;
    dropped.state = daemon_swarm_coordinator::ClientState::Dropped;
    assert!(admit(&cfg, Phase::WaitingForMembers, &[dropped], &[], &cand).is_ok());
}

#[test]
fn join_via_tick_adds_to_roster() {
    let cfg = base_config();
    let state = new_state(cfg);
    let k = key(1);
    let (state, out) = daemon_swarm_coordinator::tick(
        state,
        daemon_swarm_coordinator::Input::Message(join_msg(&k)),
    );
    assert!(out
        .iter()
        .any(|o| matches!(o, daemon_swarm_coordinator::Output::Note(_))));
    assert!(state.is_healthy_member(&peer_id(&k)));
}
