// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// TDD §3.1 PROTO-7/9/10 unit suites — the per-behavior units the P1 live drill exercised as
// behavior but never pinned as isolated unit tests (spec §6.4/§6.2/§9). PROTO-7: heartbeat
// staleness + the K-record-absence drop counter + the straggle stall-window exemption; PROTO-9:
// the epoch boundary at `epoch_rounds`; PROTO-10: deterministic single-checkpointer election.
//
// Oracle provenance (swarm-ledger-p2-b1.md): from-definition — the `tick` state machine is the
// spec's pure function; these assert its exact per-input effects.

mod common;

use common::*;
use daemon_swarm_coordinator::{tick, ClientState, CoordinatorState, Input, Notice, Output, Phase};
use daemon_swarm_proto::assignment::elect_checkpointer;
use daemon_swarm_proto::{peer_id, PeerId, Seed};

/// Drive round `round` to finalize: `committers` commit + are storage-receipted. If the committed
/// set is the full healthy roster the receipt fast-path finalizes immediately; otherwise (silent
/// peers) the train deadline clock finalizes. Returns the outputs of the tick that finalized (the
/// one carrying the `RoundRecord` + any drop notices).
fn finalize_round_timeout(
    mut state: CoordinatorState,
    committers: &[daemon_swarm_proto::SigningKey],
    coord: &daemon_swarm_proto::SigningKey,
    round: u64,
) -> (CoordinatorState, Vec<Output>) {
    for k in committers {
        let (s, _) = tick(state, Input::Message(commitment_msg(k, round, 7)));
        state = s;
    }
    let entries: Vec<(PeerId, u8)> = committers.iter().map(|k| (peer_id(k), 7)).collect();
    let (s, out) = tick(state, Input::Message(receipt_msg(coord, round, &entries)));
    state = s;
    // If the receipt already finalized (round advanced or run halted), those are the finalize outputs.
    if state.round != round || matches!(state.phase, Phase::Cooldown | Phase::Finished) {
        return (state, out);
    }
    // Otherwise a silent peer held the round open; the train deadline finalizes.
    let deadline = state.phase_start_s + state.config.round_train_max_s + 1;
    tick(state, Input::Clock(deadline))
}

// ===== PROTO-7: heartbeat staleness + K-absence drop + straggle exemption =======================

#[test]
fn peer_silent_emits_stale() {
    // A peer that neither commits nor heartbeats goes "stale": its `last_seen_round` lags the run
    // and its absence counter increments, while a present (committing) peer resets to zero. This is
    // the stale signal the drop counter (and the observer's run-health) reads.
    let mut cfg = base_config();
    cfg.k_absences = 3; // high enough that one absence does not yet drop
    cfg.epoch_rounds = 10;
    let a = key(1);
    let b = key(2);
    let c = key(3);
    let coord = key(200);
    let state = to_first_round(cfg, &[a.clone(), b.clone(), c.clone()]);

    // Round 0: a,b commit; c is silent (no commit, no heartbeat).
    let (state, _) = finalize_round_timeout(state, &[a.clone(), b.clone()], &coord, 0);
    assert_eq!(state.round, 1, "round advanced");

    let a_m = state.roster.iter().find(|m| m.peer == peer_id(&a)).unwrap();
    let c_m = state.roster.iter().find(|m| m.peer == peer_id(&c)).unwrap();
    assert_eq!(a_m.absences, 0, "a was present ⇒ absence reset");
    assert_eq!(c_m.absences, 1, "silent c accrued one absence (stale)");
    assert!(
        c_m.last_seen_round < state.round,
        "silent c's last_seen_round lags the run (stale liveness)"
    );
    assert_eq!(
        c_m.state,
        ClientState::Healthy,
        "one absence is not yet a drop"
    );
}

#[test]
fn k_absences_drops() {
    // A peer silent for exactly K record-rounds is dropped, and the drop is announced.
    let mut cfg = base_config();
    cfg.k_absences = 2;
    cfg.epoch_rounds = 10; // no epoch boundary in the way
    let a = key(1);
    let b = key(2);
    let c = key(3);
    let coord = key(200);
    let c_id = peer_id(&c);
    let mut state = to_first_round(cfg, &[a.clone(), b.clone(), c.clone()]);

    // Round 0: c silent → absence 1 (not yet dropped).
    let (s, _) = finalize_round_timeout(state, &[a.clone(), b.clone()], &coord, 0);
    state = s;
    let c_m = state.roster.iter().find(|m| m.peer == c_id).unwrap();
    assert_eq!(c_m.absences, 1);
    assert_eq!(c_m.state, ClientState::Healthy);

    // Round 1: c silent again → absence 2 == K → dropped + Notice::Dropped(c).
    let (state, out) = finalize_round_timeout(state, &[a.clone(), b.clone()], &coord, 1);
    let c_m = state.roster.iter().find(|m| m.peer == c_id).unwrap();
    assert_eq!(c_m.state, ClientState::Dropped, "K=2 absences drops c");
    assert!(
        out.iter()
            .any(|o| matches!(o, Output::Note(Notice::Dropped(p)) if *p == c_id)),
        "the drop is announced"
    );
}

#[test]
fn straggle_within_window_not_dropped() {
    // A peer that signals `Straggle` each round is exempt from the absence counter for the stall
    // window, so it survives past K silent-equivalent rounds.
    let mut cfg = base_config();
    cfg.k_absences = 2;
    cfg.epoch_rounds = 100;
    cfg.stall_rounds_max = 2;
    let a = key(1);
    let b = key(2);
    let c = key(3);
    let coord = key(200);
    let c_id = peer_id(&c);
    let mut state = to_first_round(cfg, &[a.clone(), b.clone(), c.clone()]);

    for round in 0..4 {
        // c straggles (heartbeats a stall) but never commits.
        for k in [&a, &b] {
            let (s, _) = tick(state, Input::Message(commitment_msg(k, round, 7)));
            state = s;
        }
        let entries: Vec<(PeerId, u8)> = [&a, &b].iter().map(|k| (peer_id(k), 7)).collect();
        let (s, _) = tick(state, Input::Message(receipt_msg(&coord, round, &entries)));
        state = s;
        let (s, _) = tick(state, Input::Message(straggle_msg(&c, round)));
        state = s;
        let deadline = state.phase_start_s + state.config.round_train_max_s + 1;
        let (s, _) = tick(state, Input::Clock(deadline));
        state = s;
    }
    let c_m = state.roster.iter().find(|m| m.peer == c_id).unwrap();
    assert_eq!(
        c_m.state,
        ClientState::Healthy,
        "straggling c stays healthy"
    );
    assert_eq!(
        c_m.absences, 0,
        "straggle within the window is not an absence"
    );
}

// ===== PROTO-9: epoch ends at epoch_rounds =====================================================

#[test]
fn epoch_ends_at_epoch_rounds() {
    // With `epoch_rounds = N`, the coordinator finishes exactly N committed rounds then enters
    // Cooldown at the epoch boundary (roster re-freeze), not before.
    let mut cfg = base_config();
    cfg.epoch_rounds = 3;
    cfg.stop = daemon_swarm_proto::envelope::StopCondition::Rounds(1_000);
    let a = key(1);
    let b = key(2);
    let coord = key(200);
    let mut state = to_first_round(cfg, &[a.clone(), b.clone()]);

    // Rounds 0 and 1 stay in the round loop (re-open RoundTrain).
    for round in 0..2 {
        let (s, _) = finalize_round_timeout(state, &[a.clone(), b.clone()], &coord, round);
        state = s;
        assert_eq!(
            state.phase,
            Phase::RoundTrain,
            "round {round} re-opens the loop"
        );
    }
    // Round 2 is the third committed round == epoch_rounds → Cooldown.
    let (state, _) = finalize_round_timeout(state, &[a.clone(), b.clone()], &coord, 2);
    assert_eq!(
        state.phase,
        Phase::Cooldown,
        "epoch boundary at round 3 → Cooldown"
    );
    assert_eq!(state.rounds_done, 3);
    assert_eq!(state.epoch, 0, "epoch increments only after cooldown");
}

// ===== PROTO-10: deterministic single-checkpointer election ====================================

fn pk(n: u32) -> PeerId {
    let mut b = [0u8; 32];
    b[..4].copy_from_slice(&n.to_be_bytes());
    PeerId(b)
}

#[test]
fn checkpointer_deterministic_from_seed() {
    // The election is a pure function of (seed, roster): reproducible, order-independent, and
    // seed-sensitive (a different seed generally elects a different checkpointer).
    let roster: Vec<PeerId> = (0..8).map(pk).collect();
    let mut reversed = roster.clone();
    reversed.reverse();
    let seed = Seed([0x42; 32]);

    let a = elect_checkpointer(&roster, &seed);
    let b = elect_checkpointer(&reversed, &seed);
    assert_eq!(a, b, "election is independent of roster input order");
    assert_eq!(a, elect_checkpointer(&roster, &seed), "reproducible");

    // A different seed almost always moves the winner across an 8-peer roster.
    let others: Vec<Option<PeerId>> = (0u8..16)
        .map(|s| elect_checkpointer(&roster, &Seed([s; 32])))
        .collect();
    assert!(
        others.iter().filter(|&&w| w != a).count() > 0,
        "the seed determines the winner"
    );
}

#[test]
fn elects_single_checkpointer() {
    // Exactly one checkpointer, and it is a roster member; empty roster ⇒ None.
    let roster: Vec<PeerId> = (0..5).map(pk).collect();
    let seed = Seed([7; 32]);
    let winner = elect_checkpointer(&roster, &seed).expect("non-empty roster elects one");
    assert!(
        roster.contains(&winner),
        "the checkpointer is a roster member"
    );
    assert_eq!(
        elect_checkpointer(&[], &seed),
        None,
        "empty roster ⇒ no checkpointer"
    );
    // A one-peer roster always elects that peer.
    assert_eq!(elect_checkpointer(&[pk(99)], &seed), Some(pk(99)));
}
