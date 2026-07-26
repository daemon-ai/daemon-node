// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The input-replay tier-1 lane** (refactor §5 A1→A2 acceptance; ABI companion §8.7): record
// a real v2 run, then re-drive the same module from the journal alone and assert every guest
// decision reproduces bit-for-bit — the wiring of `daemon-vhc-observe::journal::verifier` (the
// §8.7 typed contract) over `daemon-vhc-host::run::replay` (the wasm execution). This activates
// the journal-soak invariant (refactor §12.6) for v2: a recorded run IS its decisions.
//
// Coverage: the toy averager (timers, clock reads, periodic publishes) — plus the negatives: a
// tampered recorded publish is a typed `Diverged`, and a journal missing a recorded input makes
// the guest's request itself the divergence. (The compute@2 trainer's journal-replay soak lives
// in the trainer-goldens lane; data@2's in data_fetch.)

#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_vhc_host::run::{
    replay, start_run, MemorySink, ReplayEnd, ReplayScript, RunConfig, RunEnd, RunIdentity,
    SinkEntry,
};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_observe::journal::record::{
    Body, ClockRec, DeviceProfileRec, EventRec, PublishRec, ReadBackRec, Record, TerminalRec,
};
use daemon_vhc_observe::journal::verifier::{
    run_replay, ExpectedDecision, GuestUnderReplay, PayloadSource, ReplayOutcome, ReplayPlan,
    ReplayStep,
};
use daemon_vhc_proto::Hash;

fn guest(name: &str) -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm(name)
}

/// Record a v2 run through a `MemorySink`; `drive` gets the pump for staging etc., then the
/// harness waits for `publishes` frames, stops, and returns the recorded entries + the end.
/// The recording identity for one test run (shared with `verify` so the replay script's
/// `rng_seed` re-derivation sees the same identity the recording driver derived from).
fn identity_for(wasm: &[u8], instance: u64) -> RunIdentity {
    RunIdentity {
        run_id: [0xBB; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance,
        module: *blake3::hash(wasm).as_bytes(),
    }
}

fn record(
    wasm: &[u8],
    config: Vec<u8>,
    instance: u64,
    publishes: usize,
    drive: impl FnOnce(&daemon_vhc_host::run::PumpHandle),
) -> (Vec<SinkEntry>, RunEnd) {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let identity = identity_for(wasm, instance);
    let run_cfg = RunConfig::new(identity, [0x61; 32], config, Vec::new());
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();
    drive(&pump);
    let deadline = Instant::now() + Duration::from_secs(30);
    while pump.published().len() < publishes {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {publishes} publishes (have {})",
            pump.published().len()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    let end = run.wait().expect("guest thread clean");
    let entries = sink.lock().expect("sink").entries.clone();
    (entries, end)
}

/// Lift the sink mirror into §8.3 records (what the on-disk journal would hold), ordinal-ordered.
fn to_records(entries: &[SinkEntry]) -> Vec<Record> {
    let mut out = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let body = match e {
            SinkEntry::Event { at, frame } => Body::Event(EventRec {
                at: *at,
                frame: frame.clone(),
            }),
            SinkEntry::ReadBack {
                src,
                kind,
                status,
                value,
            } => Body::ReadBack(ReadBackRec {
                src: *src,
                kind: *kind,
                status: *status,
                value: Some(value.clone()),
                sidecar: None,
            }),
            SinkEntry::Clock { now } => Body::Clock(ClockRec { now: *now }),
            SinkEntry::DeviceProfile { profile } => Body::DeviceProfile(DeviceProfileRec {
                profile: profile.clone(),
            }),
            SinkEntry::Publish {
                channel,
                seq,
                payload_hash,
                frame,
            } => Body::Publish(PublishRec {
                channel: *channel,
                seq: *seq,
                hash: Hash(*payload_hash),
                frame: frame.clone(),
            }),
            SinkEntry::Terminal {
                kind,
                outcome,
                trap,
            } => Body::Terminal(TerminalRec {
                kind: *kind,
                outcome: *outcome,
                // The trap info travels into the record, context included — that is the field a
                // replay verdict compares, so dropping it here would make the verdict compare
                // nothing.
                trap: trap.as_ref().map(|(code, import, context, detail)| {
                    daemon_vhc_observe::journal::record::TrapInfo {
                        code: code.clone(),
                        import: import.clone(),
                        context: context.clone(),
                        detail: detail.clone(),
                    }
                }),
            }),
            _ => continue,
        };
        out.push(Record::new(i as u64, body));
    }
    out
}

/// The §8.7 `GuestUnderReplay` seam over the host replay's slice-attributed decisions: the run
/// already happened synchronously (`replay` — a replay can never block, every input is in
/// the journal), so delivery here walks the per-slice groups in recorded order. Ordinals are
/// clerical (journal bookkeeping the replay cannot know); decisions carry the recorded ord so
/// equality judges the substance: channel, seq, payload hash.
struct ReplayedDecisions {
    per_event: Vec<Vec<(u64, u64, [u8; 32])>>, // (channel, seq, payload_hash) per slice
    next_event: usize,
    expected_ords: Vec<u64>,
    next_ord: usize,
}

impl ReplayedDecisions {
    fn new(run: &daemon_vhc_host::run::ReplayedRun, plan: &ReplayPlan) -> Self {
        let mut per_event = vec![Vec::new(); run.events_delivered];
        for d in &run.decisions {
            per_event[d.event_index.min(run.events_delivered.saturating_sub(1))].push((
                d.channel,
                d.seq,
                d.payload_hash,
            ));
        }
        let expected_ords = plan
            .expected
            .iter()
            .map(|e| match e {
                ExpectedDecision::Publish { ord, .. } => *ord,
                _ => 0, // non_exhaustive: future decision kinds carry their own ords
            })
            .collect();
        Self {
            per_event,
            next_event: 0,
            expected_ords,
            next_ord: 0,
        }
    }
}

impl GuestUnderReplay for ReplayedDecisions {
    fn deliver_event(&mut self, _ord: u64, _at: u64, _frame: &[u8]) -> Vec<ExpectedDecision> {
        let group = self
            .per_event
            .get(self.next_event)
            .cloned()
            .unwrap_or_default();
        self.next_event += 1;
        group
            .into_iter()
            .map(|(channel, seq, hash)| {
                let ord = self
                    .expected_ords
                    .get(self.next_ord)
                    .copied()
                    .unwrap_or_default();
                self.next_ord += 1;
                ExpectedDecision::Publish {
                    ord,
                    channel,
                    seq,
                    hash: Hash(hash),
                }
            })
            .collect()
    }

    fn supply_import(&mut self, _step: &ReplayStep) {
        // Inputs were consumed by the synchronous host replay from the same journal.
    }
}

/// All Phase-A recorded read-backs are inline; a sidecar fetch would itself be a finding.
struct NoSidecars;
impl PayloadSource for NoSidecars {
    fn fetch(
        &self,
        _sref: &daemon_vhc_observe::journal::record::SidecarRef,
        _ord: u64,
    ) -> Option<Vec<u8>> {
        None
    }
}

/// Record → replay → verify: the full §8.7 wiring for one run. `instance` names the recording
/// identity (the deterministic `rng_seed` re-derivation input — the run header's job in a real
/// journal).
fn verify(
    wasm: &[u8],
    config: &[u8],
    entries: &[SinkEntry],
    instance: u64,
) -> (ReplayOutcome, ReplayEnd) {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut script = ReplayScript::from_entries(entries);
    script.identity = Some(identity_for(wasm, instance));
    let replayed = replay(&worker, wasm, config, &[], script).expect("replay harness");
    let plan = ReplayPlan::from_records(&to_records(entries));
    let mut guest = ReplayedDecisions::new(&replayed, &plan);
    let outcome = run_replay(&plan, &mut guest, &NoSidecars);
    (outcome, replayed.end)
}

/// The toy averager (timers + clock + periodic publishes): replay reproduces all three
/// publishes bit-for-bit and the recorded outcome.
#[test]
fn toy_averager_replay_reproduces_every_decision_bit_for_bit() {
    let wasm = guest("toy_averager");
    let (entries, end) = record(&wasm, vec![3u8], 1, 3, |_| {});
    assert!(matches!(end, RunEnd::Outcome(0)), "recorded end: {end:?}");

    let (outcome, replay_end) = verify(&wasm, &[3u8], &entries, 1);
    assert_eq!(replay_end, ReplayEnd::Outcome(0));
    match outcome {
        ReplayOutcome::Pass { decisions } => assert_eq!(decisions, 3),
        other => panic!("expected Pass, got {other:?}"),
    }
}

/// A tampered journal (one flipped bit in a recorded publish hash) is a typed `Diverged`
/// naming the recorded and replayed decisions — never a pass, never a panic.
#[test]
fn tampered_recorded_publish_is_a_typed_divergence() {
    let wasm = guest("toy_averager");
    let (mut entries, _) = record(&wasm, vec![3u8], 4, 3, |_| {});
    let tampered = entries
        .iter_mut()
        .find_map(|e| match e {
            SinkEntry::Publish { payload_hash, .. } => {
                payload_hash[0] ^= 1;
                Some(())
            }
            _ => None,
        })
        .is_some();
    assert!(tampered, "a publish record exists to tamper");

    let (outcome, replay_end) = verify(&wasm, &[3u8], &entries, 4);
    assert_eq!(
        replay_end,
        ReplayEnd::Outcome(0),
        "execution itself is fine"
    );
    match outcome {
        ReplayOutcome::Diverged(d) => {
            assert!(matches!(d.recorded, ExpectedDecision::Publish { .. }));
            assert!(d.replayed.is_some(), "the guest DID decide — differently");
        }
        other => panic!("expected Diverged, got {other:?}"),
    }
}

/// A journal missing a recorded input makes the guest's own request the divergence: the replay
/// ends `Diverged` (the guest armed its periodic timer, but the recorded tag-5 arms were
/// stripped).
#[test]
fn missing_recorded_input_is_a_replay_divergence() {
    let wasm = guest("toy_averager");
    let (mut entries, _) = record(&wasm, vec![3u8], 5, 3, |_| {});
    let before = entries.len();
    entries.retain(|e| !matches!(e, SinkEntry::TimerArm { .. }));
    assert!(entries.len() < before, "timer arms were recorded");

    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let mut script = ReplayScript::from_entries(&entries);
    script.identity = Some(identity_for(&wasm, 5));
    let replayed = replay(&worker, &wasm, &[3u8], &[], script).expect("replay harness");
    match replayed.end {
        ReplayEnd::Diverged(msg) => assert!(msg.contains("none recorded"), "{msg}"),
        other => panic!("expected Diverged, got {other:?}"),
    }
}
