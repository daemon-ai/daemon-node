// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **Replay-under-coalescing** (refactor §6's named tier-1 gate; ABI §4.7/§8.3/§8.7): the journal
//! records what was DELIVERED — plus every drop/coalesce as a tag-7 record — and replay must not
//! care what was dropped: re-feeding the delivered sequence reproduces every decision bit-exact
//! regardless of the advisory-queue pressure the recording ran under.
//!
//! The deterministic coalescing driver: `toy-averager`'s burst knob (config byte 2) arms K timers
//! at the same deadline, so they fire in ONE pump batch; with the declared `Timer` depth D < K,
//! the batch coalesces (drop-oldest, journaled) before the guest sees a single event — no timing
//! races, byte-reproducible.
//!
//! Dev/test harness: shells `cargo build` for the guests (the `v2_event_loop.rs` pattern), so the
//! fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use daemon_vhc_host::v2::{
    decode_event_frame, replay_v2, start_run, EventV2, MemorySink, ReplayEnd, ReplayScript, RunEnd,
    RunIdentity, SinkEntry, V2RunConfig,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.ancestors().nth(3).unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

static BUILD: Once = Once::new();

fn toy_averager_wasm() -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join("target/wasm32-unknown-unknown/release/toy_averager.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Burst 6 timers into a depth-2 advisory queue: the coalescing is deterministic (one pump
/// batch), journaled, and REPLAY DOES NOT CARE — every decision reproduces bit-exact from the
/// delivered sequence alone.
#[test]
fn replay_under_coalescing_reproduces_decisions_bit_exact() {
    const TICKS: u8 = 6;
    const BURST: u8 = 6;
    const TIMER_DEPTH: usize = 2;

    let wasm = toy_averager_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes())).expect("admitted");

    let identity = RunIdentity {
        run_id: [0xD1; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    // config: [n ticks, channel 0, burst]; the declared Timer depth is the coalescing bound.
    let config = vec![TICKS, 0u8, BURST];
    let mut run_cfg = V2RunConfig::new(identity, [0x91; 32], config.clone(), Vec::new());
    run_cfg.advisory_depth = TIMER_DEPTH;
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())).expect("start");

    // Stop intent registered at the run's OUTPUT cut (§4.4): the Stop enqueues atomically with
    // the TICKS-th publish, so the guest's last re-armed timer (armed on fold TICKS-1, still
    // pending after the final fold) can never fire into the recorded stream — an embedder-side
    // poll + stop() races that timer and loses under load.
    run.pump
        .stop_at_publishes(usize::from(TICKS), daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("register stop cut");

    // Watchdog only (the cut does the stopping): the averager reaches all TICKS folds.
    let deadline = Instant::now() + Duration::from_secs(30);
    while run.pump.published().len() < usize::from(TICKS) {
        assert!(
            Instant::now() < deadline,
            "timed out at {} publishes",
            run.pump.published().len()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(matches!(run.wait().expect("thread"), RunEnd::Outcome(0)));

    let entries: Vec<SinkEntry> = sink.lock().expect("sink").entries.clone();

    // Coalescing HAPPENED and was journaled (§4.7): the 6-timer burst into a depth-2 queue
    // deterministically drops the 4 oldest, each tag-7 with the timer's identity.
    let timer_drops: Vec<u64> = entries
        .iter()
        .filter_map(|e| match e {
            SinkEntry::Drop {
                class: 1, dropped, ..
            } => dropped.timer_id,
            _ => None,
        })
        .collect();
    assert_eq!(
        timer_drops,
        vec![1, 2, 3, 4],
        "the burst coalesces the four oldest timers, in order, journaled"
    );
    // The delivered sequence contains only the surviving timers.
    let delivered_timers = entries
        .iter()
        .filter(|e| {
            matches!(e, SinkEntry::Event { frame, .. }
                if matches!(decode_event_frame(frame), Ok(EventV2::Timer { .. })))
        })
        .count();
    assert_eq!(
        delivered_timers,
        usize::from(TICKS),
        "2 burst survivors + 4 re-armed singles reach the guest"
    );

    // THE GATE: replay from the journal alone — the drops are invisible to it (§8.7: "replay is
    // exact regardless of drops"); every decision reproduces bit-exact. Since B2 the averager
    // reads the identity-derived `rng_seed` (deterministic, re-derived at replay), so the script
    // carries the recording identity — the run header's job in a real journal.
    let mut script = ReplayScript::from_entries(&entries);
    script.identity = Some(RunIdentity {
        run_id: [0xD1; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    });
    let replayed = replay_v2(&worker, &wasm, &config, &[], script).expect("replay harness");
    assert_eq!(replayed.end, ReplayEnd::Outcome(0));
    let recorded: Vec<(u64, u64, [u8; 32])> = entries
        .iter()
        .filter_map(|e| match e {
            SinkEntry::Publish {
                channel,
                seq,
                payload_hash,
                ..
            } => Some((*channel, *seq, *payload_hash)),
            _ => None,
        })
        .collect();
    let redriven: Vec<(u64, u64, [u8; 32])> = replayed
        .decisions
        .iter()
        .map(|d| (d.channel, d.seq, d.payload_hash))
        .collect();
    assert_eq!(recorded.len(), usize::from(TICKS));
    assert_eq!(
        recorded, redriven,
        "replay must not care what was dropped — the journaled DELIVERED sequence is the whole \
         input"
    );
}
