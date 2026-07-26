// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The AUTHORING PARITY gate: the round a FLEET box actually opens, driven from the genesis the
// FLEET authoring produces — the production `coordinator_quorum.wasm` blob configured by
// `ceremony_genesis()`'s own envelope, and the round it opens planned by the trainer's own
// planner against the trainer config that same envelope embeds.
//
// Why it exists. Every other lane in the battery authors with `live_genesis` (the acceptance
// tier), and `live_genesis` takes the round window and the inner-step count as spec inputs that
// its callers have always kept consistent. So the acceptance suite proves that A genesis opens a
// runnable round — never that THE genesis a fleet box runs does. Three defects have now lived in
// exactly that gap (the corpus key space, the published-key suffix, and this one), and this is the
// first lane whose subject is the frozen authoring path itself.
//
// What it drives, end to end:
//
//   1. `ceremony_genesis()` authors the fleet genesis at the ratified ceremony parameters
//      (min = max = 3 trainers, the calibrated wall-clock timers, cadence 8 / retention 64).
//   2. `configure_coordinator` derives the coordinator seat from those frozen bytes and the REAL
//      coordinator guest runs on the envelope's VERBATIM config — the host never re-authors it.
//   3. The three roster peers Join and heartbeat ready, exactly the frame flow a fleet bring-up
//      produces, and the coordinator opens round 0.
//   4. Every roster peer plans that round through the trainer's own two SDK calls —
//      `interval_for` then `slice_interval`, the pair `plan_open_fetches` makes (tiny-llama's
//      live round path) — parameterized by the `steps_per_round`/`micro_batch`/`roster` decoded
//      out of the same envelope. Each peer must get a whole inner loop's worth of sequences, and
//      the peers' intervals must tile the opened window exactly.
//   5. The coordinator must arm a REAL timer, so the authored `*_s` deadlines are the wall-clock
//      seconds the operator calibrated rather than counts of delivered events.
//
// The bound, stated honestly. This lane runs the coordinator half as the production module and
// the trainer half as the trainer's own planning functions — it does not instantiate the trainer
// guest, because the frozen geometry's round is 786_507_264 parameters × 30 steps × 2048 tokens
// (hours of CPU, and the reason `ceremony_training_step` gates the optimizer at a shortened
// sequence). Nothing about the seam under test is geometry-dependent: the window arithmetic is
// the same functions with the same arguments the guest passes, and a schedule that plans zero
// steps here plans zero fetches there.
//
// THE RED LINE. Against the round window as it was authored before this lane existed — sized to
// the peer count instead of derived from the trainer config — every peer's interval is one
// sequence, `slice_interval` refuses an interval its 30 steps do not divide, and the peer plans
// NOTHING: the trainer sits the round out, the coordinator waits for a commitment that never
// comes, and with no real timer armed the run cannot even time out. That is precisely the park
// this gate fails on.

// Dev/test harness: the guest builder shells `cargo` for the guests workspace.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeSet;
use std::time::Duration;

use ciborium::value::Value;

use daemon_vhc_host::run::SinkEntry;
use daemon_vhc_net::PublishedArtifact;
use daemon_vhc_proto::{
    blake3_hash, peer_id, CapabilitySet, FrozenGenesis, Hash, IrohId, PeerId, SigningKey,
};
use daemon_vhc_sdk_consensus::coordinator::CoordinatorState;
use daemon_vhc_sdk_consensus::messages::{BatchWindow, Heartbeat, Join, ThroughputClass};
use daemon_vhc_sdk_consensus::VhcMessage;
use daemon_vhc_sdk_rounds::{interval_for, slice_interval, MicroWindow};
use daemon_vhc_testkit::ceremony::{
    ceremony_genesis, CeremonyGenesisSpec, CeremonyRunTimers, CEREMONY_SEQ_LEN,
};
use daemon_vhc_testkit::coordinator_config::WALL_CLOCK_TICK_PERIOD_MS;
use daemon_vhc_testkit::genesis_run::phase_a_grants;
use daemon_vhc_testkit::{configure_coordinator, Coordinator};

/// The ratified fleet membership: three certified trainer seats, floor == ceiling
/// (P2 lesson 1 — `min_peers` is the exact initial roster).
const FLEET_PEERS: usize = 3;

/// The run label the fleet authoring uses for its admission id.
const RUN_LABEL: &str = "vhc-ceremony-authored-round";

/// The calibrated fleet timers (ceremony §9), in wall-clock seconds.
const WARMUP_S: u64 = 300;
const ROUND_MAX_S: u64 = 600;
const WITNESS_S: u64 = 300;
const COOLDOWN_S: u64 = 60;
const STOP_ROUNDS: u64 = 48;

/// One fleet peer's node key (the box's certified base identity, stood in for here).
fn fleet_key(i: usize) -> SigningKey {
    SigningKey::from_bytes(blake3::hash(format!("ceremony-fleet-seat/{i}").as_bytes()).as_bytes())
}

/// Author the fleet genesis through the FROZEN path, pinning `coordinator_wasm` as the
/// coordinator module so the real blob is the one the envelope configures.
fn author_fleet_genesis(coordinator_wasm: &[u8], roster: &[PeerId]) -> FrozenGenesis {
    let author = SigningKey::from_bytes(&[0x42; 32]);
    // The corpus pins are ceremony-time inputs; this lane never fetches an object, so their
    // identities stand in. What they must NOT do is change the round arithmetic — and they cannot.
    let manifest = Hash([0xAB; 32]);
    let corpus_artifacts = vec![
        (
            "corpus-manifest.cbor".to_string(),
            PublishedArtifact::CorpusManifest(manifest),
        ),
        (
            "tokenizer.json".to_string(),
            PublishedArtifact::CorpusTokenizer(Hash([0x70; 32])),
        ),
        (
            "shard-0.bin".to_string(),
            PublishedArtifact::CorpusShard(Hash([0x01; 32])),
        ),
    ];
    let spec = CeremonyGenesisSpec {
        run_label: RUN_LABEL,
        coordinator_module: blake3_hash(coordinator_wasm),
        trainer_module: Hash([0x7A; 32]),
        corpus_manifest: manifest,
        corpus_artifacts: &corpus_artifacts,
        seq_len: u64::from(CEREMONY_SEQ_LEN),
        trusted_bases: roster,
        roster,
        upgrade_authority: Vec::new(),
        min_peers: FLEET_PEERS as u32,
        max_peers: FLEET_PEERS as u32,
        remote_ckpt_cadence_rounds: 8,
        payload_retention_rounds: 64,
        timers: CeremonyRunTimers {
            warmup_s: WARMUP_S,
            round_max_s: ROUND_MAX_S,
            witness_s: WITNESS_S,
            cooldown_s: COOLDOWN_S,
            stop_rounds: STOP_ROUNDS,
        },
    };
    ceremony_genesis(&spec, &author).expect("the fleet genesis authors")
}

/// One field of an opaque role config.
fn field(config: &Value, name: &str) -> Value {
    let Value::Map(entries) = config else {
        panic!("a role config is a map");
    };
    entries
        .iter()
        .find_map(|(k, v)| matches!(k, Value::Text(t) if t == name).then(|| v.clone()))
        .unwrap_or_else(|| panic!("`{name}` in the role config"))
}

fn uint_field(config: &Value, name: &str) -> u64 {
    u64::try_from(i128::from(
        field(config, name).as_integer().expect("an integer field"),
    ))
    .expect("a non-negative field")
}

#[test]
fn the_fleet_authoring_opens_a_round_its_own_trainer_config_can_plan() {
    let coordinator_wasm = daemon_vhc_guest_build::guest_wasm("coordinator_quorum");
    let keys: Vec<SigningKey> = (0..FLEET_PEERS).map(fleet_key).collect();
    let roster: Vec<PeerId> = keys.iter().map(peer_id).collect();

    let frozen = author_fleet_genesis(&coordinator_wasm, &roster);
    let env = frozen.decode().expect("decode the fleet genesis");

    // -- the trainer's half of the round, read out of the envelope (never transcribed) ----------
    let trainer_config = &env.roles.get("trainer").expect("trainer role").config;
    let steps_per_round =
        u32::try_from(uint_field(trainer_config, "steps_per_round")).expect("step count");
    let micro_batch = u32::try_from(uint_field(trainer_config, "micro_batch")).expect("micro");
    let Value::Array(config_roster) = field(trainer_config, "roster") else {
        panic!("the trainer roster is an array");
    };
    let planning_roster: Vec<PeerId> = config_roster
        .iter()
        .map(|v| match v {
            Value::Bytes(b) => PeerId(<[u8; 32]>::try_from(b.as_slice()).expect("a peer id")),
            other => panic!("a roster entry is bytes, got {other:?}"),
        })
        .collect();
    assert_eq!(planning_roster, roster, "the genesis assigns to the fleet");

    // -- the coordinator's half: the production blob, on the envelope's verbatim config ---------
    let spec = configure_coordinator(&frozen).expect("the genesis configures a coordinator");
    let mut coord = Coordinator::start(
        &coordinator_wasm,
        &spec,
        phase_a_grants(),
        0,
        keys[0].to_bytes(),
    )
    .expect("the coordinator blob starts under the fleet genesis");

    // The bring-up frame flow: every seat joins, then heartbeats model-ready.
    for key in &keys {
        coord
            .deliver(
                key,
                &VhcMessage::Join(Join {
                    run_id: RUN_LABEL.to_string(),
                    iroh_id: IrohId([0x44; 32]),
                    class: ThroughputClass::C1,
                    capabilities: CapabilitySet::new(),
                    envelope_hash: None,
                }),
            )
            .expect("deliver a join");
    }
    for key in &keys {
        coord
            .deliver(
                key,
                &VhcMessage::Heartbeat(Heartbeat {
                    round: 0,
                    ready: Some(true),
                }),
            )
            .expect("deliver a readiness heartbeat");
    }

    let opened = match coord
        .next_decision(Duration::from_secs(60))
        .expect("the coordinator opens round 0")
        .2
    {
        VhcMessage::RoundOpen(ro) => ro,
        other => panic!("the fleet coordinator's first decision must be RoundOpen, got {other:?}"),
    };
    assert_eq!(opened.round, 0);

    // -- the seam: can the peers this round is assigned to actually plan it? --------------------
    //
    // `plan_open_fetches` calls exactly these two functions, in this order, with exactly these
    // arguments (tiny-llama's live round path). A peer that plans no inner steps issues no
    // `data_fetch`, stages no batches, and never completes the round it was opened for.
    let expected_per_peer = u64::from(steps_per_round) * u64::from(micro_batch);
    let mut covered: BTreeSet<u64> = BTreeSet::new();
    for peer in &planning_roster {
        let interval = interval_for(opened.batch, opened.seed, &planning_roster, peer);
        let steps = slice_interval(interval, steps_per_round, micro_batch);
        assert_eq!(
            steps.len(),
            steps_per_round as usize,
            "peer {} plans {} inner steps over its {}-sequence interval [{}, {}) against \
             steps_per_round {steps_per_round}: an interval the inner loop cannot divide plans \
             ZERO fetches, so this peer would sit the round out and never commit",
            peer.to_hex(),
            steps.len(),
            interval.end - interval.start,
            interval.start,
            interval.end,
        );
        let sequences: u64 = steps
            .iter()
            .flat_map(|s| s.micro.iter())
            .map(|m: &MicroWindow| m.end - m.start)
            .sum();
        assert_eq!(
            sequences,
            expected_per_peer,
            "peer {} trains {sequences} sequences, not the {expected_per_peer} its config \
             schedules",
            peer.to_hex(),
        );
        covered.extend(interval.start..interval.end);
    }
    let window = opened.batch.end - opened.batch.start;
    assert_eq!(
        window,
        expected_per_peer * planning_roster.len() as u64,
        "the opened window is the roster's worth of the trainer's own inner loop",
    );
    assert_eq!(
        covered,
        (opened.batch.start..opened.batch.end).collect::<BTreeSet<u64>>(),
        "the peers' intervals must tile the opened window exactly — no sequence trained twice, \
         none skipped",
    );

    // -- the clock the authored deadlines are counted on ----------------------------------------
    //
    // The guest arms a real timer only when its config carries a non-zero `tick_period_ms`;
    // otherwise it advances one synthetic tick per delivered event and every `*_s` deadline is a
    // count of events that a quiet run never reaches.
    let coordinator_config = &env
        .roles
        .get("coordinator")
        .expect("coordinator role")
        .config;
    assert_eq!(
        uint_field(coordinator_config, "tick_period_ms"),
        WALL_CLOCK_TICK_PERIOD_MS,
    );
    let armed: Vec<u64> = coord
        .sink_entries()
        .iter()
        .filter_map(|e| match e {
            SinkEntry::TimerArm { delay, .. } => Some(*delay),
            _ => None,
        })
        .collect();
    assert!(
        armed.contains(&WALL_CLOCK_TICK_PERIOD_MS),
        "the fleet coordinator must arm its real {WALL_CLOCK_TICK_PERIOD_MS} ms timer — the \
         authored warmup/round/witness/cooldown walls are wall-clock seconds only if something \
         measures wall time; armed timers: {armed:?}",
    );

    // …and those walls are the calibrated ones, in the unit the timer gives them.
    let state: CoordinatorState = field(coordinator_config, "state")
        .deserialized()
        .expect("the coordinator state decodes");
    assert_eq!(
        (
            state.config.warmup_s,
            state.config.round_train_max_s,
            state.config.round_witness_s,
            state.config.cooldown_s,
        ),
        (WARMUP_S, ROUND_MAX_S, WITNESS_S, COOLDOWN_S),
    );
    assert_eq!(
        opened.batch,
        BatchWindow {
            start: 0,
            end: window
        },
    );

    coord.stop().expect("the coordinator stops clean");
}
