// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The REAL-GEOMETRY guest-init gate: the production trainer guest brings its state plane up at the
// FROZEN fleet-ceremony geometry (`daemon_vhc_testkit::ceremony` — 786_507_264 parameters,
// seq 2048, the seed-form state contract with the pinned `expected_root`) under the production
// sandbox budgets, on the CPU lane.
//
// Why this suite exists (the class it locks down): every other guest lane in the battery runs a
// TOY geometry — the acceptance tier trains the 64-dim structural model, the trainer goldens run
// tiny profiles — so an init path whose cost scales with the geometry passes all of them and dies
// on the fleet. That is exactly what happened: the fresh-join seed-init materialized the whole
// master family (and a zeroed `ef` family) as guest-resident `Vec<Vec<f32>>`, ~2.93 GiB per copy at
// this geometry, which no wasm32 linear memory can hold — the trainer guest trapped
// `GuestPanic`/`unreachable` ~51 ms into its role session, before fetching a single corpus shard.
//
// The gate is deliberately INIT-ONLY (no rounds, no corpus, no device training): it starts the run,
// queues Stop immediately, and lets the guest observe it at its FIRST `next_event` — which it only
// reaches once the whole init has streamed. So a clean `Outcome(0)` here IS the proof that init
// completed, and it carries the guest's own seed-init cross-check with it (the guest asserts its
// sealed master fold against the genesis `expected_root`, so a drifting expansion traps instead of
// passing). The engine profile is `EngineConfig::real_model` — the SAME budgets
// `daemon-vhc-worker`'s join engine runs, INCLUDING the unraised 64 MiB linear-memory cap, so the
// bounded-guest-memory invariant (design §3.2: the guest folds at O(chunks in flight)) is what is
// actually under test.
//
// Cost: this is the battery's one real-geometry lane. It expands + folds ~2.93 GiB of state through
// the guest in 4 MiB windows and allocates the real fp32 device working set host-side (master +
// both AdamW moments ≈ 8.8 GiB of ndarray tensors, plus the sealed master family in the state
// store). Keep it single, keep it init-only.

// Dev/test harness: the guest builder shells `cargo` for the guests workspace.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};

use daemon_vhc_host::run::{start_run, MemorySink, RunConfig, RunEnd, RunIdentity};
use daemon_vhc_host::{BackendKind, EngineConfig, Worker};
use daemon_vhc_proto::det_state::family_byte_len;
use daemon_vhc_proto::{to_canonical_vec, PeerId};
use daemon_vhc_testkit::ceremony::{
    ceremony_param_numels, ceremony_state_chunk_size, ceremony_trainer_config_harness,
    CEREMONY_PARAM_COUNT,
};

/// The sole trainer identity the harness form pins as `peer`/`roster`.
const PEER: [u8; 32] = [0x3b; 32];

#[test]
fn ceremony_geometry_trainer_init_streams_under_the_production_budgets() {
    let wasm = daemon_vhc_guest_build::guest_wasm("tiny_llama");
    let numels = ceremony_param_numels();
    let family_bytes = family_byte_len(&numels.iter().map(|&n| n as u64).collect::<Vec<_>>());
    assert_eq!(
        numels.iter().map(|&n| n as u64).sum::<u64>(),
        CEREMONY_PARAM_COUNT,
        "the harness drives the frozen ceremony geometry"
    );

    // The production join-lane profile — notably NOT a raised linear-memory cap (see the module
    // docs): a conforming guest streams its families, so the toy-tier 64 MiB cap must suffice at
    // any geometry.
    let engine = EngineConfig::real_model(BackendKind::Cpu, None);
    assert_eq!(
        engine.max_memory_bytes,
        EngineConfig::default().max_memory_bytes,
        "the real-model profile must not buy its way past the bounded-guest-memory invariant"
    );
    let worker = Worker::new(engine).expect("engine");

    let identity = RunIdentity {
        run_id: [0xce; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let cfg_bytes = to_canonical_vec(&ceremony_trainer_config_harness(&[PeerId(PEER)]))
        .expect("ceremony trainer config (harness form)");
    let mut run_cfg = RunConfig::new(identity, [0x9d; 32], cfg_bytes, Vec::new());
    run_cfg.state_chunk_size = ceremony_state_chunk_size();
    run_cfg.compute_queue_depth = 1 << 20;

    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink)).expect("start");
    let pump = run.pump.clone();
    // Queued now, observed by the guest only at its first `next_event` — i.e. after init.
    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!(
            "ceremony-geometry init must complete inside the production sandbox budgets, got \
             {other:?} (a GuestPanic here is a guest-resident family or size arithmetic that does \
             not scale to the fleet geometry; OutOfFuel is the real-model fuel budget)"
        ),
    }

    // The state plane the init left behind: the self-sealed master family ([SF-R1]) plus the
    // zeroed `ef` family. `ef` is all zeros, so its full-length chunks dedup to a single chunk
    // object in the store — retained bytes are the master family plus `ef`'s short per-parameter
    // tails, never a second full family.
    let stats = pump.state_store_stats();
    assert_eq!(
        stats.sealed_folds, 2,
        "fresh-join init seals exactly the master + ef families, got {stats:?}"
    );
    assert!(
        stats.retained_bytes >= family_bytes,
        "the sealed master family must retain the full {family_bytes} B image, got {stats:?}"
    );
    eprintln!(
        "ceremony_geometry: init sealed {} folds, {} retained bytes (master family {} B) at \
         {} parameters / {} B windows",
        stats.sealed_folds,
        stats.retained_bytes,
        family_bytes,
        CEREMONY_PARAM_COUNT,
        ceremony_state_chunk_size(),
    );
}
