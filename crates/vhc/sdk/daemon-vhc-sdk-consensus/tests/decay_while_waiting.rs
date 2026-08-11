// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Decay-while-waiting (reliability spec §7, the C2 floor-breach membership wedge).
//!
//! Absence accounting otherwise lives only inside round finalization, so once the floor breaches
//! there are no rounds, hence no decay, hence a dead member holds its roster seat forever. These
//! scenarios pin the repair: during `WaitingForMembers`, a healthy member not heard for the full
//! round-scaled absence equivalent (`k_absences × (round_train_max_s + round_witness_s)`) decays
//! without a round — including TWO dead trainers at once, the C2 shape one churn slot cannot
//! absorb — while a member that keeps heartbeating (or whose window simply has not elapsed)
//! keeps its seat. Consensus safety: decay only shrinks the healthy set; the freed seats are
//! re-filled through the ordinary Join path.

mod common;

use common::*;
use daemon_vhc_proto::peer_id;
use daemon_vhc_sdk_consensus::coordinator::{
    tick, ClientState, CoordinatorState, Input, Notice, Output, Phase,
};

/// Drive the run into `WaitingForMembers` with a breached floor: a, b, c train; b and c go
/// silent and are dropped by round accounting; the floor breach parks the run waiting. Returns
/// the waiting state with `a` the only healthy member.
fn breached_floor() -> (
    CoordinatorState,
    daemon_vhc_proto::SigningKey,
    daemon_vhc_proto::SigningKey,
    daemon_vhc_proto::SigningKey,
) {
    let mut cfg = base_config();
    cfg.min_peers = 3;
    cfg.max_peers = 4;
    cfg.k_absences = 2;
    cfg.epoch_rounds = 100;
    let (a, b, c) = (key(1), key(2), key(3));
    let coord = key(200);
    let mut state = to_first_round(cfg, &[a.clone(), b.clone(), c.clone()]);
    // Rounds 0 and 1: only a commits; b and c accrue absences to k=2 and drop at round 1's
    // finalize, breaching the floor (healthy 1 < min 3) → Cooldown → WaitingForMembers.
    for round in 0..2 {
        let (s, _) = tick(state, Input::Message(commitment_msg(&a, round, 7)));
        state = s;
        let entries = vec![(peer_id(&a), 7u8)];
        let (s, _) = tick(state, Input::Message(receipt_msg(&coord, round, &entries)));
        state = s;
        let deadline = state.phase_start_s + state.config.round_train_max_s + 1;
        let (s, _) = tick(state, Input::Clock(deadline));
        state = s;
    }
    assert_eq!(state.phase, Phase::Cooldown, "the floor breach cools down");
    let cooled = state.now_s + state.config.cooldown_s;
    let (s, _) = tick(state, Input::Clock(cooled));
    state = s;
    assert_eq!(state.phase, Phase::WaitingForMembers, "the run waits");
    (state, a, b, c)
}

#[test]
fn two_dead_trainers_decay_while_waiting_and_the_rejoiners_heal_the_floor() {
    // The two-dead-trainer wedge, healed without a round: b and c are already dead (dropped by
    // round accounting before the breach; their hosts stay down). A live trainer a heartbeats
    // through the wait. d joins during the wait and then dies silently — a zombie seat that,
    // pre-repair, NO mechanism could ever reclaim: decay lived only in round finalization and
    // the waiting phase has no rounds (the C2 wedge).
    let (mut state, a, _b, _c) = breached_floor();
    let d = key(4);
    let (s, _) = tick(state, Input::Message(join_msg(&d)));
    state = s;
    assert_eq!(state.healthy_count(), 2, "a + d");

    // The wait stretches: mid-window the clock advances (no decay yet — 200 < 260) and a
    // heartbeats, re-stamping its liveness; d never speaks again. Past the window (k=2 ×
    // (train 100 + witness 30) = 260 s from d's join), d decays WITHOUT a round.
    let t0 = state.now_s;
    let (s, out) = tick(state, Input::Clock(t0 + 200));
    state = s;
    assert!(
        !out.iter()
            .any(|o| matches!(o, Output::Note(Notice::Dropped(_)))),
        "no decay inside the window"
    );
    let (s, _) = tick(state, Input::Message(heartbeat_msg(&a, 2)));
    state = s;
    let (s, out) = tick(state, Input::Clock(t0 + 261));
    state = s;
    let d_id = peer_id(&d);
    assert!(
        out.iter()
            .any(|o| matches!(o, Output::Note(Notice::Dropped(p)) if *p == d_id)),
        "the zombie's decay is announced"
    );
    let d_m = state.roster.iter().find(|m| m.peer == d_id).unwrap();
    assert_eq!(d_m.state, ClientState::Dropped, "d decayed while waiting");
    let a_m = state.roster.iter().find(|m| m.peer == peer_id(&a)).unwrap();
    assert_eq!(
        a_m.state,
        ClientState::Healthy,
        "the heartbeating member keeps its seat"
    );

    // The freed seats admit the replacements through the ordinary Join path; the floor gathers
    // and the run warms up again — the wedge is healed without a single round.
    let (e, f) = (key(5), key(6));
    let (s, _) = tick(state, Input::Message(join_msg(&e)));
    state = s;
    let (s, _) = tick(state, Input::Message(join_msg(&f)));
    state = s;
    assert_eq!(state.healthy_count(), 3, "a + e + f gather the floor");
    let gathered = state.now_s + 1;
    let (state, _) = tick(state, Input::Clock(gathered));
    assert_eq!(
        state.phase,
        Phase::Warmup,
        "the healed floor leaves WaitingForMembers"
    );
}

#[test]
fn the_decay_window_is_floored_at_the_phase_start_and_reset_by_liveness() {
    // (1) No premature decay: a restored/reconstructed roster (last_seen_s = 0 via serde
    // default) is NOT mass-dropped at the first waiting tick — staleness is floored at the
    // phase start, so the full window must elapse IN this phase first.
    let (mut state, a, _b, _c) = breached_floor();
    for m in &mut state.roster {
        m.last_seen_s = 0; // the restored-snapshot shape
    }
    let just_short = state.phase_start_s + 259;
    let (s, out) = tick(state, Input::Clock(just_short));
    state = s;
    assert!(
        !out.iter()
            .any(|o| matches!(o, Output::Note(Notice::Dropped(_)))),
        "no decay before the window elapses in-phase"
    );
    let a_id = peer_id(&a);
    assert!(
        state
            .roster
            .iter()
            .any(|m| m.peer == a_id && m.is_healthy()),
        "the survivor still holds its seat"
    );

    // (2) Liveness resets the window: a heartbeat inside the window re-stamps the member, so
    // the original deadline passes without decaying it.
    let (s, _) = tick(state, Input::Message(heartbeat_msg(&a, 2)));
    state = s;
    let past_original_window = state.phase_start_s + 261;
    let (state, _) = tick(state, Input::Clock(past_original_window));
    assert!(
        state
            .roster
            .iter()
            .any(|m| m.peer == a_id && m.is_healthy()),
        "a heard member never decays at the stale deadline"
    );
}
