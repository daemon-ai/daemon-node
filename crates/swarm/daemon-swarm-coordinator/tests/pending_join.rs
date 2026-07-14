// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! §6.2 pending-join sharp edge (Merge-2 adjudication (d); TDD PROTO-2/§6.2 gap register).
//!
//! Joins take effect only at epoch boundaries because the roster is frozen in `Warmup` (spec §6.2).
//! A join that arrives while the run is still `WaitingForMembers` is applied **roster-direct**; a
//! join that arrives **after** the `WaitingForMembers → Warmup` transition is staged `pending` and
//! materializes only at the next epoch boundary — so with `epoch_rounds = 0` (a single-epoch run) a
//! late join **never** materializes mid-run. This is the operational note the ledger adds to §6.2:
//! declared-run authors MUST set `min_peers` = the expected initial roster size, or workers racing
//! the warmup transition are staged until an epoch boundary that (for `epoch_rounds = 0`) never
//! comes. The `ws_live_workers` gate harness encodes exactly this (`min_peers == NUM_WORKERS`).
//!
//! These pin the `tick` routing (`tick.rs::on_join`: `WaitingForMembers` → `upsert_member`, else
//! `state.pending.push`) + the epoch-boundary drain, so the sharp edge cannot silently regress.

mod common;

use common::*;
use daemon_swarm_proto::peer_id;

use daemon_swarm_coordinator::{tick, Input, Phase};

fn keys(n: u8) -> Vec<daemon_swarm_proto::SigningKey> {
    (1..=n).map(key).collect()
}

/// Drive one full round to completion via the commit + storage-receipt fast path (no timeouts).
fn complete_round(
    mut state: daemon_swarm_coordinator::CoordinatorState,
    ks: &[daemon_swarm_proto::SigningKey],
    coord: &daemon_swarm_proto::SigningKey,
    round: u64,
    seed: u8,
) -> daemon_swarm_coordinator::CoordinatorState {
    for k in ks {
        let (s, _) = tick(state, Input::Message(commitment_msg(k, round, seed)));
        state = s;
    }
    let entries: Vec<(daemon_swarm_proto::PeerId, u8)> =
        ks.iter().map(|k| (peer_id(k), seed)).collect();
    let (s, _) = tick(state, Input::Message(receipt_msg(coord, round, &entries)));
    s
}

#[test]
fn join_in_waiting_for_members_is_roster_direct() {
    // The initial roster forms while the run is `WaitingForMembers`: each join lands directly in the
    // roster as a healthy member, nothing is staged pending.
    let mut cfg = base_config();
    cfg.min_peers = 3; // don't leave WaitingForMembers on the first two joins
    let mut state = new_state(cfg);
    assert_eq!(state.phase, Phase::WaitingForMembers);

    for k in &keys(2) {
        let (s, _) = tick(state, Input::Message(join_msg(k)));
        state = s;
    }
    assert_eq!(
        state.phase,
        Phase::WaitingForMembers,
        "still gathering members"
    );
    assert_eq!(state.healthy_count(), 2, "both joined the roster directly");
    assert!(
        state.pending.is_empty(),
        "nothing is staged while gathering"
    );
    assert!(state.is_healthy_member(&pid(1)) && state.is_healthy_member(&pid(2)));
}

#[test]
fn join_after_warmup_transition_is_staged_pending() {
    // With `min_peers = 2`, two joins open the run (Warmup → RoundTrain). A THIRD join then arrives
    // mid-run: it is staged `pending`, not added to the live roster, and does not change the healthy
    // count (the roster is frozen for the epoch, §6.2).
    let mut cfg = base_config();
    cfg.min_peers = 2;
    cfg.epoch_rounds = 100; // long epoch: no boundary interrupts this test
    let state = to_first_round(cfg, &keys(2));
    assert_eq!(state.healthy_count(), 2);

    let late = key(3);
    let (state, out) = tick(state, Input::Message(join_msg(&late)));
    // Admitted (auth passed) …
    assert!(
        out.iter().any(|o| matches!(
            o,
            daemon_swarm_coordinator::Output::Note(daemon_swarm_coordinator::Notice::Admitted(_))
        )),
        "a mid-run join that passes admission is Admitted (as pending)"
    );
    // … but staged pending, NOT a live roster member.
    assert_eq!(
        state.healthy_count(),
        2,
        "the live roster is frozen for the epoch"
    );
    assert!(
        !state.is_healthy_member(&peer_id(&late)),
        "the mid-run joiner is not a live member"
    );
    assert!(
        state.pending.iter().any(|m| m.peer == peer_id(&late)),
        "the mid-run joiner is staged pending"
    );
}

#[test]
fn pending_join_materializes_at_epoch_boundary() {
    // A join staged pending in-epoch becomes a healthy roster member only after the epoch boundary
    // (Cooldown → WaitingForMembers, epoch++). Here `epoch_rounds = 2`, so the boundary arrives.
    let mut cfg = base_config();
    cfg.min_peers = 2;
    cfg.epoch_rounds = 2;
    cfg.stop = daemon_swarm_proto::envelope::StopCondition::Rounds(100);
    let coord = key(200);
    let ks = keys(2);
    let mut state = to_first_round(cfg.clone(), &ks);

    // Stage a mid-run join in round 0.
    let late = key(3);
    let (s, _) = tick(state, Input::Message(join_msg(&late)));
    state = s;
    assert!(
        state.pending.iter().any(|m| m.peer == peer_id(&late)),
        "staged pending mid-run"
    );

    // Complete the epoch (rounds 0 and 1) → Cooldown.
    for round in 0..2 {
        state = complete_round(state, &ks, &coord, round, (round + 1) as u8);
    }
    assert_eq!(state.phase, Phase::Cooldown, "epoch end → Cooldown");
    assert_eq!(state.epoch, 0);

    // Cooldown timeout → WaitingForMembers (epoch 1): pending drains into the roster.
    let now = state.now_s + cfg.cooldown_s + 1;
    let (state, _) = tick(state, Input::Clock(now));
    assert_eq!(state.phase, Phase::WaitingForMembers);
    assert_eq!(state.epoch, 1, "new epoch");
    assert!(
        state.is_healthy_member(&peer_id(&late)),
        "the pending join materialized at the epoch boundary"
    );
    assert!(
        state.pending.is_empty(),
        "the pending queue drained at the boundary"
    );
}

#[test]
fn pending_join_never_materializes_mid_run_when_epoch_rounds_zero() {
    // THE sharp edge (adjudication (d)): with `epoch_rounds = 0` (a single epoch for the whole run)
    // there is no mid-run epoch boundary, so a join arriving after warmup stays pending forever — it
    // never materializes while the run trains. This is why a declared-run author MUST set
    // `min_peers` = the expected initial roster: a worker that races the warmup transition would
    // otherwise be stranded pending for the entire run.
    let mut cfg = base_config();
    cfg.min_peers = 2;
    cfg.epoch_rounds = 0; // single epoch — no mid-run roster refreeze
    cfg.stop = daemon_swarm_proto::envelope::StopCondition::Rounds(100);
    let coord = key(200);
    let ks = keys(2);
    let mut state = to_first_round(cfg, &ks);

    let late = key(3);
    let (s, _) = tick(state, Input::Message(join_msg(&late)));
    state = s;
    assert!(state.pending.iter().any(|m| m.peer == peer_id(&late)));

    // Train several rounds: no epoch boundary is ever crossed, so the joiner stays pending.
    for round in 0..6 {
        state = complete_round(state, &ks, &coord, round, (round + 1) as u8);
        assert_eq!(
            state.phase,
            Phase::RoundTrain,
            "still training (no epoch end)"
        );
        assert!(
            !state.is_healthy_member(&peer_id(&late)),
            "the late joiner never materializes mid-run (round {round})"
        );
        assert!(
            state.pending.iter().any(|m| m.peer == peer_id(&late)),
            "the late joiner is still stranded pending (round {round})"
        );
    }
    assert_eq!(
        state.healthy_count(),
        2,
        "roster unchanged for the whole run"
    );
}
