// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The commit rule + committee-math seams (spec §6.4 I6, §9, §12; TDD PROTO-5/6/10/15/17).

mod common;

use common::*;
use daemon_swarm_proto::messages::{
    AttestEntry, Attestation, BatchWindow, Commitment, Locator, RecordEntry, ThroughputClass,
};
use daemon_swarm_proto::{
    commit_set, elect_checkpointer, select_verifiers, Hash, IrohId, PeerId, Seed,
};

use daemon_swarm_coordinator::commit::{
    all_evidenced, committed_entries, has_evidence, quorum_root,
};
use daemon_swarm_coordinator::epoch::{ready_to_update_epoch, EpochInputs, EpochTrigger};
use daemon_swarm_coordinator::state::{Member, RoundState};
use daemon_swarm_coordinator::{tick, Input, Output, Rejection};

fn member(seed: u8) -> Member {
    Member::joining(pid(seed), IrohId([seed; 32]), ThroughputClass::C2, 0)
}

fn commitment(round: u64, seed: u8) -> Commitment {
    Commitment {
        round,
        payload: payload_hash(seed),
        size: 1_000,
        locators: vec![Locator::StoreKey("k".to_string())],
    }
}

fn slot_with(witnesses: Vec<PeerId>) -> RoundState {
    RoundState::opened(
        0,
        Seed([1; 32]),
        0,
        BatchWindow { start: 0, end: 100 },
        witnesses,
    )
}

// ----- PROTO-5: commit rule as a pure function of signed evidence -----

#[test]
fn proto5_receipt_evidence_admits_commitment() {
    let roster = vec![member(1), member(2)];
    let mut rs = slot_with(vec![]);
    rs.commitments.insert(pid(1), commitment(0, 10));
    rs.commitments.insert(pid(2), commitment(0, 20));
    // Storage receipt covers both.
    rs.receipts.push(RecordEntry {
        peer: pid(1),
        hash: payload_hash(10),
        size: 1_000,
    });
    rs.receipts.push(RecordEntry {
        peer: pid(2),
        hash: payload_hash(20),
        size: 1_000,
    });

    let entries = committed_entries(&rs, &roster);
    assert_eq!(entries.len(), 2);
    assert!(all_evidenced(&rs, &roster));
    // Ordered by peer bytes (I3).
    let sorted = {
        let mut e = entries.clone();
        e.sort_by_key(|x| x.peer);
        e
    };
    assert_eq!(entries, sorted);
}

#[test]
fn proto5_missing_evidence_holds_the_commit() {
    let roster = vec![member(1), member(2)];
    let mut rs = slot_with(vec![]);
    rs.commitments.insert(pid(1), commitment(0, 10));
    rs.commitments.insert(pid(2), commitment(0, 20));
    // Only peer 1 has a receipt.
    rs.receipts.push(RecordEntry {
        peer: pid(1),
        hash: payload_hash(10),
        size: 1_000,
    });

    let entries = committed_entries(&rs, &roster);
    assert_eq!(entries.len(), 1, "unevidenced commitment is held out");
    assert_eq!(entries[0].peer, pid(1));
    assert!(!all_evidenced(&rs, &roster));
    // The record root commits to exactly the evidenced set.
    let pairs: Vec<(PeerId, Hash)> = entries.iter().map(|e| (e.peer, e.hash)).collect();
    assert_eq!(commit_set(&pairs).commitment().count, 1);
}

// ----- PROTO-6: witness-quorum gate on the attestation evidence path -----

#[test]
fn proto6_witness_quorum_gate() {
    let witnesses: Vec<PeerId> = (10..14).map(pid).collect(); // 4 witnesses → quorum 3
    let mut rs = slot_with(witnesses.clone());
    rs.commitments.insert(pid(1), commitment(0, 30));

    let cover = vec![AttestEntry {
        peer: pid(1),
        hash: payload_hash(30),
    }];
    let attest = |w_inline: &[AttestEntry]| Attestation {
        round: 0,
        set: commit_set(
            &w_inline
                .iter()
                .map(|e| (e.peer, e.hash))
                .collect::<Vec<_>>(),
        )
        .commitment(),
        inline: Some(w_inline.to_vec()),
    };

    // 2 of 4 witnesses cover → below quorum(4)=3 → no evidence.
    rs.attestations.insert(witnesses[0], attest(&cover));
    rs.attestations.insert(witnesses[1], attest(&cover));
    assert!(!has_evidence(&rs, &pid(1), &payload_hash(30)));

    // A third witness reaches quorum → evidence.
    rs.attestations.insert(witnesses[2], attest(&cover));
    assert!(has_evidence(&rs, &pid(1), &payload_hash(30)));
}

// ----- §6.4 root-only attestation coverage (Wave 3) -----

fn bare_root_attest(pairs: &[(PeerId, Hash)]) -> Attestation {
    Attestation {
        round: 0,
        set: commit_set(pairs).commitment(),
        inline: None,
    }
}

fn inline_attest(pairs: &[(PeerId, Hash)]) -> Attestation {
    Attestation {
        round: 0,
        set: commit_set(pairs).commitment(),
        inline: Some(
            pairs
                .iter()
                .map(|(peer, hash)| AttestEntry {
                    peer: *peer,
                    hash: *hash,
                })
                .collect(),
        ),
    }
}

#[test]
fn root_only_attestation_covers_at_quorum() {
    let witnesses: Vec<PeerId> = (10..14).map(pid).collect(); // 4 witnesses → quorum 3
    let set_pairs = vec![(pid(1), payload_hash(30))];
    let root = commit_set(&set_pairs).commitment().root;

    let mut rs = slot_with(witnesses.clone());
    rs.commitments.insert(pid(1), commitment(0, 30));

    // Three witnesses attest a BARE root R (inline omitted) — a quorum agrees on the root …
    for w in &witnesses[..3] {
        rs.attestations.insert(*w, bare_root_attest(&set_pairs));
    }
    assert_eq!(quorum_root(&rs), Some(root), "3/4 agree on the same root");
    // … but root agreement alone does not pin membership without an opening or receipt.
    assert!(!has_evidence(&rs, &pid(1), &payload_hash(30)));

    // The fourth witness supplies the inline opening of R → membership pinned exactly → covered.
    rs.attestations
        .insert(witnesses[3], inline_attest(&set_pairs));
    assert!(has_evidence(&rs, &pid(1), &payload_hash(30)));
}

#[test]
fn root_only_below_quorum_not_covered() {
    let witnesses: Vec<PeerId> = (10..14).map(pid).collect(); // quorum 3
    let mut rs = slot_with(witnesses.clone());
    rs.commitments.insert(pid(1), commitment(0, 30));

    let set_pairs = vec![(pid(1), payload_hash(30))];
    let other_pairs = vec![(pid(2), payload_hash(99))];
    // Only 2 witnesses agree on R (an inline opening + a bare root); the other 2 attest a different
    // root — so no root reaches the quorum of 3.
    rs.attestations
        .insert(witnesses[0], inline_attest(&set_pairs));
    rs.attestations
        .insert(witnesses[1], bare_root_attest(&set_pairs));
    rs.attestations
        .insert(witnesses[2], bare_root_attest(&other_pairs));
    rs.attestations
        .insert(witnesses[3], bare_root_attest(&other_pairs));

    assert_eq!(quorum_root(&rs), None, "no root reaches quorum");
    assert!(!has_evidence(&rs, &pid(1), &payload_hash(30)));
}

#[test]
fn root_only_agreement_without_opening_is_not_evidence() {
    let witnesses: Vec<PeerId> = (10..14).map(pid).collect();
    let set_pairs = vec![(pid(1), payload_hash(30))];
    let mut rs = slot_with(witnesses.clone());
    rs.commitments.insert(pid(1), commitment(0, 30));

    // All four witnesses attest the same bare root — full agreement, but nobody opens the set and
    // there is no StorageReceipt, so the payload cannot enter the record (§6.4: an opening or a
    // receipt must pin the specific member).
    for w in &witnesses {
        rs.attestations.insert(*w, bare_root_attest(&set_pairs));
    }
    assert!(quorum_root(&rs).is_some());
    assert!(!has_evidence(&rs, &pid(1), &payload_hash(30)));
    assert!(committed_entries(&rs, &[member(1)]).is_empty());
}

// ----- §A.3 small-n witness-quorum edges in the commit rule -----

#[test]
fn small_n_attestation_quorum_covers_at_the_special_case_boundary() {
    // The §A.3 adopted small-n quorum specials (⌈⅔n⌉ with 1→1, 2→2, 3→2) must hold in the *commit
    // rule*, not just the pure `witness_quorum` helper: a payload is covered by the attestation path
    // exactly when a quorum of witnesses supplies an inline opening of its set. Below the quorum it
    // is held out; at the quorum it is admitted. Pins the small rosters the gate's ≥4-peer run and
    // any degraded (churned-down) epoch pass through.
    use daemon_swarm_proto::assignment::witness_quorum;

    for n in 1u32..=3 {
        let witnesses: Vec<PeerId> = (10..10 + n as u8).map(pid).collect();
        let quorum = witness_quorum(n) as usize;
        assert_eq!(
            quorum,
            match n {
                1 => 1,
                2 => 2,
                3 => 2,
                _ => unreachable!(),
            },
            "small-n special quorum for n={n}"
        );

        let set_pairs = vec![(pid(1), payload_hash(30))];
        let mut rs = slot_with(witnesses.clone());
        rs.commitments.insert(pid(1), commitment(0, 30));

        // One short of quorum: not covered.
        for w in witnesses.iter().take(quorum - 1) {
            rs.attestations.insert(*w, inline_attest(&set_pairs));
        }
        assert!(
            !has_evidence(&rs, &pid(1), &payload_hash(30)),
            "n={n}: {} of {quorum} witnesses is below quorum",
            quorum - 1
        );

        // Exactly quorum witnesses open the set → covered.
        rs.attestations
            .insert(witnesses[quorum - 1], inline_attest(&set_pairs));
        assert!(
            has_evidence(&rs, &pid(1), &payload_hash(30)),
            "n={n}: a quorum of {quorum} witnesses covers the commitment"
        );
    }
}

#[test]
fn single_peer_round_finalizes_on_self_evidence() {
    // n=1 degenerate roster: a one-peer round is `all_evidenced` once its single member is covered
    // (quorum(1)=1 — the sole witness's inline opening, or equivalently a StorageReceipt). The commit
    // rule must not deadlock at n=1 (the smallest churn floor a gate epoch can shrink to).
    let roster = vec![member(1)];
    let mut rs = slot_with(vec![pid(1)]);
    rs.commitments.insert(pid(1), commitment(0, 42));
    assert!(!all_evidenced(&rs, &roster), "no evidence yet");

    rs.attestations
        .insert(pid(1), inline_attest(&[(pid(1), payload_hash(42))]));
    assert!(
        all_evidenced(&rs, &roster),
        "the sole peer's self-attestation (quorum 1) evidences the round"
    );
    let entries = committed_entries(&rs, &roster);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].peer, pid(1));
}

// ----- PROTO-5: bad signature rejected at the frame -----

#[test]
fn proto5_bad_signature_rejected() {
    let ks: Vec<_> = (1..=2u8).map(key).collect();
    let state = to_first_round(base_config(), &ks);
    let mut msg = commitment_msg(&ks[0], 0, 9);
    msg.sig.0[0] ^= 0xff; // tamper
    let (_, out) = tick(state, Input::Message(msg));
    assert_eq!(out, vec![Output::Reject(Rejection::BadSignature)]);
}

// ----- PROTO-15: verifier committee is a no-op at 0% -----

#[test]
fn proto15_verifier_noop_at_zero_percent() {
    let roster: Vec<PeerId> = (1..=8).map(pid).collect();
    assert!(select_verifiers(&roster, &Seed([2; 32]), 0).is_empty());
    let sampled = select_verifiers(&roster, &Seed([2; 32]), 50);
    assert_eq!(sampled.len(), 4, "ceil(8 * 50 / 100)");
    // Deterministic + distinct.
    assert_eq!(sampled, select_verifiers(&roster, &Seed([2; 32]), 50));
}

// ----- PROTO-10: deterministic checkpointer election -----

#[test]
fn proto10_checkpointer_deterministic_and_single() {
    let roster: Vec<PeerId> = (1..=6).map(pid).collect();
    let mut reversed = roster.clone();
    reversed.reverse();
    let a = elect_checkpointer(&roster, &Seed([3; 32]));
    let b = elect_checkpointer(&reversed, &Seed([3; 32]));
    assert!(a.is_some());
    assert_eq!(a, b, "order-independent + deterministic");
    assert!(roster.contains(&a.unwrap()));
    assert_eq!(elect_checkpointer(&[], &Seed([3; 32])), None);
}

// ----- PROTO-17: epoch-advance disjuncts (hivemind port) -----

#[test]
fn proto17_epoch_advance_disjuncts() {
    // batch target reached
    assert_eq!(
        ready_to_update_epoch(&EpochInputs {
            rounds_this_epoch: 3,
            epoch_rounds: 3,
            peer_epoch_lead: 0,
            eta_rounds_remaining: 10,
        }),
        EpochTrigger::BatchTarget
    );
    // a peer already leads into a later epoch
    assert_eq!(
        ready_to_update_epoch(&EpochInputs {
            rounds_this_epoch: 1,
            epoch_rounds: 3,
            peer_epoch_lead: 1,
            eta_rounds_remaining: 10,
        }),
        EpochTrigger::GlobalLead
    );
    // ETA exhausted
    assert_eq!(
        ready_to_update_epoch(&EpochInputs {
            rounds_this_epoch: 1,
            epoch_rounds: 3,
            peer_epoch_lead: 0,
            eta_rounds_remaining: 0,
        }),
        EpochTrigger::Eta
    );
    // not ready
    assert_eq!(
        ready_to_update_epoch(&EpochInputs {
            rounds_this_epoch: 1,
            epoch_rounds: 3,
            peer_epoch_lead: 0,
            eta_rounds_remaining: 5,
        }),
        EpochTrigger::None
    );
}
