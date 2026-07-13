// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Coordinator lifecycle scenarios (spec §6.2/§6.4; TDD PROTO-1/2/3/7/9/10/14).

mod common;

use common::*;
use daemon_swarm_proto::envelope::{GlobalBatch, StopCondition};
use daemon_swarm_proto::messages::SwarmMessage;
use daemon_swarm_proto::{peer_id, to_canonical_vec, PeerId, SwarmProtoVersion};

use daemon_swarm_coordinator::{
    tick, ClientState, ControlAction, CoordinatorState, Input, Notice, Output, Phase,
};

fn keys(n: u8) -> Vec<daemon_swarm_proto::SigningKey> {
    (1..=n).map(key).collect()
}

/// Drive one full round to completion via the commit + storage-receipt fast path (no timeouts).
fn complete_round(
    state: CoordinatorState,
    ks: &[daemon_swarm_proto::SigningKey],
    coord: &daemon_swarm_proto::SigningKey,
    round: u64,
    seed: u8,
) -> (CoordinatorState, Vec<Output>) {
    let mut state = state;
    for k in ks {
        let (s, _) = tick(state, Input::Message(commitment_msg(k, round, seed)));
        state = s;
    }
    let entries: Vec<(PeerId, u8)> = ks.iter().map(|k| (peer_id(k), seed)).collect();
    tick(state, Input::Message(receipt_msg(coord, round, &entries)))
}

// ----- PROTO-1: purity / determinism -----

#[test]
fn proto1_tick_is_pure_same_input_same_output() {
    let ks = keys(2);
    let state = to_first_round(base_config(), &ks);

    let input = Input::Message(commitment_msg(&ks[0], 0, 9));
    let (s1, o1) = tick(state.clone(), input.clone());
    let (s2, o2) = tick(state.clone(), input);

    assert_eq!(o1, o2, "identical inputs must yield identical outputs");
    assert_eq!(
        to_canonical_vec(&s1).unwrap(),
        to_canonical_vec(&s2).unwrap(),
        "identical inputs must yield byte-identical state"
    );
    // And the input did not mutate the original state (no hidden mutation).
    assert_ne!(
        to_canonical_vec(&state).unwrap(),
        to_canonical_vec(&s1).unwrap()
    );
}

// ----- PROTO-2: phase-timeout ladder -----

#[test]
fn proto2_phase_timeouts_walk_the_ladder() {
    let ks = keys(2);
    let mut state = new_state(base_config());
    for k in &ks {
        let (s, _) = tick(state, Input::Message(join_msg(k)));
        state = s;
    }
    assert_eq!(state.phase, Phase::WaitingForMembers);

    // min_peers met → Warmup.
    let (s, _) = tick(state, Input::Clock(1));
    state = s;
    assert_eq!(state.phase, Phase::Warmup);

    // warmup_s = 10 → RoundTrain at t=11.
    let (s, out) = tick(state, Input::Clock(11));
    state = s;
    assert_eq!(state.phase, Phase::RoundTrain);
    assert!(
        publishes(&out)
            .iter()
            .any(|m| matches!(m, SwarmMessage::RoundOpen(_))),
        "opening a round publishes RoundOpen"
    );

    // No commitments: train_max = 100 → RoundWitness at t=111.
    let (s, _) = tick(state, Input::Clock(111));
    state = s;
    assert_eq!(state.phase, Phase::RoundWitness);

    // witness = 30 → finalize (empty record) + open round 1 at t=141.
    let (s, out) = tick(state, Input::Clock(141));
    state = s;
    assert_eq!(state.phase, Phase::RoundTrain);
    assert_eq!(state.round, 1);
    assert!(publishes(&out)
        .iter()
        .any(|m| matches!(m, SwarmMessage::RoundRecord(_))));
}

// ----- PROTO-3: stored-round ring + cursor threading -----

#[test]
fn proto3_ring_wraps_and_cursor_threads() {
    let mut cfg = base_config();
    cfg.epoch_rounds = 100; // don't let an epoch boundary interrupt the ring walk
    cfg.global_batch = GlobalBatch {
        start: 100,
        end: 100,
        ramp_rounds: 0,
    };
    let ks = keys(2);
    let coord = key(200);
    let mut state = to_first_round(cfg, &ks);

    for round in 0..5 {
        assert_eq!(
            state.data_index,
            round * 100,
            "cursor threads by global_batch"
        );
        let (s, _) = complete_round(state, &ks, &coord, round, (round + 1) as u8);
        state = s;
    }
    // After completing rounds 0..=4 we are opening round 5; the ring slot for round 4 (index 0)
    // must hold round 4 (it wrapped over round 0).
    assert_eq!(state.round, 5);
    assert_eq!(state.data_index, 500);
    assert_eq!(
        state.rounds.slots[0].round, 4,
        "ring reused slot 0 for round 4"
    );
}

// ----- PROTO-9: ramp / stop / epoch -----

#[test]
fn proto9_global_batch_ramps_the_cursor() {
    let mut cfg = base_config();
    cfg.epoch_rounds = 100;
    cfg.global_batch = GlobalBatch {
        start: 100,
        end: 200,
        ramp_rounds: 4,
    };
    let ks = keys(2);
    let coord = key(200);
    let mut state = to_first_round(cfg, &ks);

    let (s, _) = complete_round(state, &ks, &coord, 0, 1);
    state = s;
    assert_eq!(state.data_index, 100, "gb(0) = 100");
    let (s, _) = complete_round(state, &ks, &coord, 1, 2);
    state = s;
    assert_eq!(state.data_index, 225, "gb(1) = 125");
}

#[test]
fn proto9_stop_tokens_finishes_run() {
    let mut cfg = base_config();
    cfg.epoch_rounds = 100;
    cfg.seq_len = 1;
    cfg.stop = StopCondition::Tokens(250); // gb = 100/round → round 2 crosses 250
    let ks = keys(2);
    let coord = key(200);
    let mut state = to_first_round(cfg, &ks);

    for round in 0..3 {
        let (s, _) = complete_round(state, &ks, &coord, round, (round + 1) as u8);
        state = s;
    }
    assert_eq!(state.phase, Phase::Cooldown, "stop → Cooldown");

    // Cooldown timeout → Finished.
    let now = state.now_s + cfg_cooldown() + 1;
    let (s, out) = tick(state, Input::Clock(now));
    assert_eq!(s.phase, Phase::Finished);
    assert!(out
        .iter()
        .any(|o| matches!(o, Output::Note(Notice::Finished))));
}

fn cfg_cooldown() -> u64 {
    base_config().cooldown_s
}

#[test]
fn proto9_epoch_boundary_returns_to_waiting() {
    let mut cfg = base_config();
    cfg.epoch_rounds = 2;
    cfg.stop = StopCondition::Rounds(100);
    let ks = keys(2);
    let coord = key(200);
    let mut state = to_first_round(cfg.clone(), &ks);

    for round in 0..2 {
        let (s, _) = complete_round(state, &ks, &coord, round, (round + 1) as u8);
        state = s;
    }
    assert_eq!(state.phase, Phase::Cooldown, "epoch end → Cooldown");
    assert_eq!(state.epoch, 0);

    let now = state.now_s + cfg.cooldown_s + 1;
    let (s, _) = tick(state, Input::Clock(now));
    assert_eq!(s.phase, Phase::WaitingForMembers);
    assert_eq!(s.epoch, 1, "new epoch after cooldown");
}

// ----- PROTO-14: halted states + pause/resume authorization -----

#[test]
fn proto14_halted_states_error() {
    for halted in [Phase::Uninitialized, Phase::Finished, Phase::Paused] {
        let mut state = new_state(base_config());
        state.phase = halted;
        let (_, out) = tick(state.clone(), Input::Clock(5));
        assert_eq!(
            out,
            vec![Output::Reject(daemon_swarm_coordinator::Rejection::Halted(
                halted
            ))],
            "clock in {halted:?} must error"
        );
        let (_, out) = tick(state, Input::Message(heartbeat_msg(&key(1), 0)));
        assert!(matches!(
            out.as_slice(),
            [Output::Reject(daemon_swarm_coordinator::Rejection::Halted(
                _
            ))]
        ));
    }
}

#[test]
fn proto14_pause_requires_authorized_principal() {
    let admin = key(250);
    let mut cfg = base_config();
    cfg.authorized = vec![peer_id(&admin)];
    let ks = keys(2);
    let state = to_first_round(cfg, &ks);

    // Unauthorized pause rejected.
    let (state, out) = tick(state, Input::Control(control(&ks[0], ControlAction::Pause)));
    assert_eq!(
        out,
        vec![Output::Reject(
            daemon_swarm_coordinator::Rejection::Unauthorized
        )]
    );
    assert_ne!(state.phase, Phase::Paused);

    // Authorized pause accepted.
    let (state, _) = tick(state, Input::Control(control(&admin, ControlAction::Pause)));
    assert_eq!(state.phase, Phase::Paused);

    // Messages are halted while paused.
    let (state, out) = tick(state, Input::Message(heartbeat_msg(&ks[0], 0)));
    assert!(matches!(
        out.as_slice(),
        [Output::Reject(daemon_swarm_coordinator::Rejection::Halted(
            Phase::Paused
        ))]
    ));

    // Unauthorized resume rejected; authorized resume returns to WaitingForMembers.
    let (state, out) = tick(
        state,
        Input::Control(control(&ks[0], ControlAction::Resume)),
    );
    assert_eq!(
        out,
        vec![Output::Reject(
            daemon_swarm_coordinator::Rejection::Unauthorized
        )]
    );
    let (state, _) = tick(
        state,
        Input::Control(control(&admin, ControlAction::Resume)),
    );
    assert_eq!(state.phase, Phase::WaitingForMembers);
}

// ----- PROTO-7 / PROTO-10: drops, straggle window, rejoin -----

/// Drive a round to completion by timeout (A/B commit + receipt; C behaves per `c_action`).
fn timeout_round_without_c(
    mut state: CoordinatorState,
    ab: &[daemon_swarm_proto::SigningKey],
    coord: &daemon_swarm_proto::SigningKey,
    round: u64,
    c_straggle: Option<&daemon_swarm_proto::SigningKey>,
) -> CoordinatorState {
    for k in ab {
        let (s, _) = tick(state, Input::Message(commitment_msg(k, round, 7)));
        state = s;
    }
    let entries: Vec<(PeerId, u8)> = ab.iter().map(|k| (peer_id(k), 7)).collect();
    let (s, _) = tick(state, Input::Message(receipt_msg(coord, round, &entries)));
    state = s;
    if let Some(c) = c_straggle {
        let (s, _) = tick(state, Input::Message(straggle_msg(c, round)));
        state = s;
    }
    // Train timeout finalizes (all committed peers are evidenced; C absent/straggling).
    let deadline = state.phase_start_s + state.config.round_train_max_s + 1;
    let (s, _) = tick(state, Input::Clock(deadline));
    s
}

#[test]
fn proto7_k_absences_drops_and_proto10_rejoin() {
    let mut cfg = base_config();
    cfg.k_absences = 2;
    cfg.epoch_rounds = 2;
    cfg.min_peers = 2;
    let a = key(1);
    let b = key(2);
    let c = key(3);
    let coord = key(200);
    let ab = [a.clone(), b.clone()];
    let mut state = to_first_round(cfg.clone(), &[a.clone(), b.clone(), c.clone()]);
    assert_eq!(state.healthy_count(), 3);

    // Round 0 and 1: C never commits → 2 absences → dropped at round 1; round 1 also ends the epoch.
    state = timeout_round_without_c(state, &ab, &coord, 0, None);
    assert_eq!(state.round, 1);
    state = timeout_round_without_c(state, &ab, &coord, 1, None);

    let c_id = peer_id(&c);
    let c_member = state.roster.iter().find(|m| m.peer == c_id).unwrap();
    assert_eq!(
        c_member.state,
        ClientState::Dropped,
        "C dropped after K=2 absences"
    );
    assert_eq!(
        state.phase,
        Phase::Cooldown,
        "round 1 also hit the epoch boundary"
    );

    // Cooldown → WaitingForMembers (epoch 1); C rejoins into the roster.
    let now = state.now_s + cfg.cooldown_s + 1;
    let (mut state, _) = tick(state, Input::Clock(now));
    assert_eq!(state.phase, Phase::WaitingForMembers);
    let (s, out) = tick(state, Input::Message(join_msg(&c)));
    state = s;
    assert!(out
        .iter()
        .any(|o| matches!(o, Output::Note(Notice::Admitted(_)))));
    let c_member = state.roster.iter().find(|m| m.peer == c_id).unwrap();
    assert_eq!(c_member.state, ClientState::Healthy, "C rejoined healthy");
}

#[test]
fn proto7_straggle_within_window_not_dropped() {
    let mut cfg = base_config();
    cfg.k_absences = 2;
    cfg.epoch_rounds = 100;
    cfg.stall_rounds_max = 2;
    let a = key(1);
    let b = key(2);
    let c = key(3);
    let coord = key(200);
    let ab = [a.clone(), b.clone()];
    let mut state = to_first_round(cfg, &[a.clone(), b.clone(), c.clone()]);

    // C never commits but heartbeats a Straggle every round → no absence counts.
    for round in 0..4 {
        state = timeout_round_without_c(state, &ab, &coord, round, Some(&c));
    }
    let c_id = peer_id(&c);
    let c_member = state.roster.iter().find(|m| m.peer == c_id).unwrap();
    assert_eq!(
        c_member.state,
        ClientState::Healthy,
        "straggling C stays healthy across the stall window"
    );
    assert_eq!(c_member.absences, 0);
}

// ----- version gate (PROTO-13 via the frame) -----

#[test]
fn message_with_wrong_version_rejected() {
    let ks = keys(2);
    let state = new_state(base_config());
    let (_, out) = tick(
        state,
        Input::Message(join_msg_version(&ks[0], SwarmProtoVersion(999))),
    );
    assert!(matches!(
        out.as_slice(),
        [Output::Reject(
            daemon_swarm_coordinator::Rejection::VersionMismatch { .. }
        )]
    ));
}

// ----- Warmup early-exit on peer readiness (Wave-3 additive) -----

#[test]
fn warmup_early_exits_when_all_ready() {
    let ks = keys(2);
    let mut state = new_state(base_config());
    for k in &ks {
        let (s, _) = tick(state, Input::Message(join_msg(k)));
        state = s;
    }
    // Entering Warmup still needs one clock (warmup_s = 10); the *exit* is what gains the early path.
    let (s, _) = tick(state, Input::Clock(1));
    state = s;
    assert_eq!(state.phase, Phase::Warmup);
    let warmup_start = state.phase_start_s;

    // One peer ready is not enough.
    let (s, _) = tick(state, Input::Message(ready_heartbeat_msg(&ks[0], 0)));
    state = s;
    assert_eq!(
        state.phase,
        Phase::Warmup,
        "one ready is not a quorum of readiness"
    );

    // Both ready → round 0 opens immediately (event-driven, no warmup timeout).
    let (s, out) = tick(state, Input::Message(ready_heartbeat_msg(&ks[1], 0)));
    state = s;
    assert_eq!(state.phase, Phase::RoundTrain);
    assert_eq!(state.round, 0);
    assert!(
        state.now_s < warmup_start + base_config().warmup_s,
        "opened before the warmup timeout would have fired"
    );
    assert!(publishes(&out)
        .iter()
        .any(|m| matches!(m, SwarmMessage::RoundOpen(_))));
}

#[test]
fn warmup_falls_back_to_timeout_without_all_ready() {
    let ks = keys(2);
    let mut state = new_state(base_config());
    for k in &ks {
        let (s, _) = tick(state, Input::Message(join_msg(k)));
        state = s;
    }
    let (s, _) = tick(state, Input::Clock(1));
    state = s;
    // Only one peer signals readiness → the early-exit gate stays closed.
    let (s, _) = tick(state, Input::Message(ready_heartbeat_msg(&ks[0], 0)));
    state = s;
    assert_eq!(state.phase, Phase::Warmup);
    // The warmup timeout still opens the round (unchanged back-compat path).
    let (s, out) = tick(state, Input::Clock(base_config().warmup_s + 2));
    state = s;
    assert_eq!(state.phase, Phase::RoundTrain);
    assert!(publishes(&out)
        .iter()
        .any(|m| matches!(m, SwarmMessage::RoundOpen(_))));
}

// ----- happy path: a fully-evidenced round records the committed set -----

#[test]
fn happy_round_records_committed_set() {
    let ks = keys(2);
    let coord = key(200);
    let state = to_first_round(base_config(), &ks);
    let (state, out) = complete_round(state, &ks, &coord, 0, 5);

    let records: Vec<_> = publishes(&out)
        .into_iter()
        .filter_map(|m| match m {
            SwarmMessage::RoundRecord(r) => Some(r.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].set.count, 2, "both peers committed + evidenced");
    assert!(records[0].drops.is_empty());
    assert_eq!(state.round, 1, "advanced to the next round");
}
