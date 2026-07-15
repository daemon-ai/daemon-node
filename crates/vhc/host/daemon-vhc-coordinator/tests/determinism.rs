// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Replay + determinism (spec §6.4 I1, §11.2; TDD PROTO-16/20).

mod common;

use common::*;
use daemon_swarm_proto::{from_canonical_slice, peer_id, to_canonical_vec, PeerId};

use daemon_swarm_coordinator::{tick, CoordinatorState, Input, Output};
use proptest::prelude::*;

/// A full scripted run as an input sequence: joins → warmup → three complete rounds.
fn scripted_inputs() -> Vec<Input> {
    let ks: Vec<_> = (1..=2u8).map(key).collect();
    let coord = key(200);
    let mut inputs = vec![
        Input::Message(join_msg(&ks[0])),
        Input::Message(join_msg(&ks[1])),
        Input::Clock(1),  // → Warmup
        Input::Clock(20), // → RoundTrain (round 0)
    ];
    for round in 0..3u64 {
        for k in &ks {
            inputs.push(Input::Message(commitment_msg(k, round, (round + 1) as u8)));
        }
        let entries: Vec<(PeerId, u8)> =
            ks.iter().map(|k| (peer_id(k), (round + 1) as u8)).collect();
        inputs.push(Input::Message(receipt_msg(&coord, round, &entries)));
    }
    inputs
}

fn run(mut state: CoordinatorState, inputs: &[Input]) -> (CoordinatorState, Vec<Vec<Output>>) {
    let mut trace = Vec::new();
    for inp in inputs {
        let (s, o) = tick(state, inp.clone());
        state = s;
        trace.push(o);
    }
    (state, trace)
}

// ----- PROTO-20: same input sequence → byte-identical state + identical outputs -----

#[test]
fn proto20_run_is_byte_reproducible() {
    let mut cfg = base_config();
    cfg.epoch_rounds = 100;
    let inputs = scripted_inputs();

    let (s1, t1) = run(new_state(cfg.clone()), &inputs);
    let (s2, t2) = run(new_state(cfg), &inputs);

    assert_eq!(t1, t2, "output traces must match");
    assert_eq!(
        to_canonical_vec(&s1).unwrap(),
        to_canonical_vec(&s2).unwrap(),
        "final state must be byte-identical"
    );
}

// ----- PROTO-20: state survives a canonical-CBOR round trip mid-run (the resync oracle) -----

#[test]
fn proto20_state_survives_cbor_round_trip() {
    let mut cfg = base_config();
    cfg.epoch_rounds = 100;
    let inputs = scripted_inputs();
    let split = inputs.len() / 2;

    // Reference: run straight through.
    let (reference, _) = run(new_state(cfg.clone()), &inputs);

    // Resumed: run half, serialize → deserialize, run the rest.
    let (mid, _) = run(new_state(cfg), &inputs[..split]);
    let bytes = to_canonical_vec(&mid).unwrap();
    let resumed: CoordinatorState = from_canonical_slice(&bytes).unwrap();
    assert_eq!(mid, resumed, "round trip is lossless");
    let (finished, _) = run(resumed, &inputs[split..]);

    assert_eq!(
        to_canonical_vec(&reference).unwrap(),
        to_canonical_vec(&finished).unwrap(),
        "resumed run reproduces the reference state byte-for-byte"
    );
}

// ----- PROTO-16: integer-only determinism under arbitrary (integer) input scripts -----

/// A compact, integer-only command the proptest maps onto tick inputs (no floats anywhere).
#[derive(Clone, Copy, Debug)]
enum Cmd {
    Tick(u64),
    Commit(u8, u8),
    Receipt(u8),
}

fn cmd_strategy() -> impl Strategy<Value = Cmd> {
    prop_oneof![
        (1u64..50).prop_map(Cmd::Tick),
        (0u8..2, 0u8..4).prop_map(|(p, s)| Cmd::Commit(p, s)),
        (0u8..4).prop_map(Cmd::Receipt),
    ]
}

fn apply(
    state: CoordinatorState,
    cmd: Cmd,
    ks: &[daemon_swarm_proto::SigningKey],
    coord: &daemon_swarm_proto::SigningKey,
) -> CoordinatorState {
    let round = state.round;
    let input = match cmd {
        Cmd::Tick(dt) => Input::Clock(state.now_s + dt),
        Cmd::Commit(p, s) => Input::Message(commitment_msg(&ks[(p as usize) % ks.len()], round, s)),
        Cmd::Receipt(s) => {
            let entries: Vec<(PeerId, u8)> = ks.iter().map(|k| (peer_id(k), s)).collect();
            Input::Message(receipt_msg(coord, round, &entries))
        }
    };
    tick(state, input).0
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn proto16_tick_is_integer_deterministic(cmds in prop::collection::vec(cmd_strategy(), 0..40)) {
        let mut cfg = base_config();
        cfg.epoch_rounds = 100;
        let ks: Vec<_> = (1..=2u8).map(key).collect();
        let coord = key(200);

        let base = to_first_round(cfg, &ks);

        let mut a = base.clone();
        let mut b = base;
        for c in &cmds {
            a = apply(a, *c, &ks, &coord);
            b = apply(b, *c, &ks, &coord);
        }
        prop_assert_eq!(to_canonical_vec(&a).unwrap(), to_canonical_vec(&b).unwrap());
    }
}
