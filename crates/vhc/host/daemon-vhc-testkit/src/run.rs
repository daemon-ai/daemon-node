// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The whole-run harness: run a production wasm blob under the real host event-loop driver with
//! simulated capability providers, journal it, then §8.7 input-replay it and assert bit-for-bit
//! decision reproduction. This generalizes the A2 t2 join-run's inline replay soak (refactor §12.6)
//! into reusable testkit infrastructure.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_vhc_host::v2::{
    replay_v2, start_run, MemorySink, ReplayEnd, ReplayScript, RunEnd, RunIdentity, SinkEntry,
    V2RunConfig,
};
use daemon_vhc_host::{select_driver, Worker};

/// One recorded/replayed decision key: `(channel, seq, payload_hash)`.
pub type Decision = (u64, u64, [u8; 32]);

/// How a run is set up under the testkit driver.
pub struct RunSpec {
    /// The frozen execution identity (§8.1) the run + journal are keyed by.
    pub identity: RunIdentity,
    /// The per-run signing key seed (the §12.1 signer).
    pub key_seed: [u8; 32],
    /// The admitted config bytes handed to `da_init` (byte-identical on replay, §9.4 step 11).
    pub config: Vec<u8>,
    /// The admitted grants bytes (byte-identical on replay).
    pub grants: Vec<u8>,
    /// Wait until the module has published at least this many frames, then stop it (a self-driven
    /// module's natural drain point). `0` runs until the module stops itself or the timeout.
    pub expect_publishes: usize,
    /// Hard wall so a wedged module cannot hang the gate.
    pub timeout: Duration,
}

impl RunSpec {
    /// A spec for a self-driven (coordinator-less) module: run until it has published
    /// `expect_publishes` frames, then stop.
    #[must_use]
    pub fn self_driven(
        identity: RunIdentity,
        key_seed: [u8; 32],
        config: Vec<u8>,
        grants: Vec<u8>,
        expect_publishes: usize,
    ) -> Self {
        Self {
            identity,
            key_seed,
            config,
            grants,
            expect_publishes,
            timeout: Duration::from_secs(60),
        }
    }
}

/// The §8.7 replay verdict for a run.
#[derive(Debug, Clone)]
pub struct ReplayReport {
    /// How the replayed run ended (`Outcome`/`Diverged`/`Trapped`/`InitRefused`).
    pub end: ReplayEnd,
    /// Every decision the replay re-derived, in order.
    pub decisions: Vec<Decision>,
    /// True iff the replayed decisions exactly equal the recorded publishes AND the replay ended in
    /// an `Outcome` — the bit-for-bit reproduction the gate requires (refactor §12.6, ABI §8.7).
    pub matched: bool,
}

/// The observable product of a whole run.
#[derive(Debug)]
pub struct WholeRunReport {
    /// How the live run ended.
    pub end: RunEnd,
    /// The publishes recorded in the journal, in order.
    pub recorded_publishes: Vec<Decision>,
    /// The §8.7 replay verdict over that journal.
    pub replay: ReplayReport,
}

impl WholeRunReport {
    /// True iff the run ended cleanly (`Outcome`) AND the recorded journal replayed bit-for-bit.
    #[must_use]
    pub fn is_green(&self) -> bool {
        matches!(self.end, RunEnd::Outcome(_)) && self.replay.matched
    }
}

/// Drive `wasm` (a production major-2 blob) under the real host event-loop driver with a simulated
/// capability substrate, journal every §8 observation, stop it cleanly, then §8.7 input-replay the
/// recorded journal and compare decisions.
///
/// # Errors
/// A `String` on any harness-level failure (non-v2 module, sandbox/start error, guest-thread
/// panic, or the module never reaching `expect_publishes` before the timeout). Guest-level endings
/// (clean outcome, trap) are carried in [`WholeRunReport::end`]; the replay verdict is
/// [`WholeRunReport::replay`].
pub fn whole_run(worker: &Worker, wasm: &[u8], spec: RunSpec) -> Result<WholeRunReport, String> {
    let module_hash = *blake3::hash(wasm).as_bytes();
    let sel = select_driver(worker, wasm, Some(&module_hash))
        .map_err(|e| format!("driver selection: {e}"))?;
    if sel.driver != daemon_vhc_abi::CandidateDriver::V2 {
        return Err(format!(
            "whole_run expects a major-2 production blob, got {:?}",
            sel.driver
        ));
    }

    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run_cfg = V2RunConfig::new(
        spec.identity.clone(),
        spec.key_seed,
        spec.config.clone(),
        spec.grants.clone(),
    );
    let run = start_run(worker, wasm, run_cfg, Box::new(sink.clone()))
        .map_err(|e| format!("start_run: {e}"))?;
    let pump = run.pump.clone();

    // Stop intent at the run's output cut (§4.4): the Stop enqueues atomically with the final
    // expected publish, so a still-armed guest timer can never fire into the recorded stream
    // after the run's natural completion (a poll + stop() would race it).
    pump.stop_at_publishes(
        spec.expect_publishes,
        daemon_vhc_abi::STOP_REASON_RUN_COMPLETE,
    )
    .map_err(|e| format!("register stop cut: {e}"))?;

    // Watchdog: the self-driven module reaches its publishes (it arms its own timers; the
    // driver's clock fires them), bounded so a wedged guest cannot hang the gate.
    let deadline = Instant::now() + spec.timeout;
    while pump.published().len() < spec.expect_publishes {
        if Instant::now() >= deadline {
            let _ = pump.stop(daemon_vhc_abi::STOP_REASON_FAULT);
            let _ = run.wait();
            return Err(format!(
                "timed out waiting for {} publishes (got {})",
                spec.expect_publishes,
                pump.published().len()
            ));
        }
        std::thread::sleep(Duration::from_millis(5));
    }

    let end = run.wait().map_err(|e| format!("guest thread: {e}"))?;

    // The recorded journal → the decisions the run made.
    let entries: Vec<SinkEntry> = sink.lock().expect("sink lock").entries.clone();
    let recorded_publishes = publishes(&entries);

    // §8.7: re-drive the recorded journal and compare every decision. The identity rides along
    // (the tag-0 run header in a real journal): `sys@2::rng_seed` re-derives from it at replay
    // (the merged toy-averager is abi 2.1 and reads its identity-derived seed).
    let mut script = ReplayScript::from_entries(&entries);
    script.identity = Some(spec.identity.clone());
    let replayed = replay_v2(worker, wasm, &spec.config, &spec.grants, script)
        .map_err(|e| format!("replay harness: {e}"))?;
    let decisions: Vec<Decision> = replayed
        .decisions
        .iter()
        .map(|d| (d.channel, d.seq, d.payload_hash))
        .collect();
    let matched = decisions == recorded_publishes && matches!(replayed.end, ReplayEnd::Outcome(_));

    Ok(WholeRunReport {
        end,
        recorded_publishes,
        replay: ReplayReport {
            end: replayed.end,
            decisions,
            matched,
        },
    })
}

/// Extract the ordered `(channel, seq, payload_hash)` publishes from a recorded journal.
fn publishes(entries: &[SinkEntry]) -> Vec<Decision> {
    entries
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
        .collect()
}
