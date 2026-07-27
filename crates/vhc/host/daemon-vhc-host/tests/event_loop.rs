// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The A2 event-loop end-to-end proof (refactor §5 A2 acceptance; ABI §13 conformance rows):
// the timer-driven `toy-averager` guest — a NON-ROUND topology using only the declared Phase-A
// subset (timers + publish) — runs under the real major-2 driver with zero host changes:
// selection admits it (majors flipped to [1,2] in this commit), `da_init`/`da_run` dispatch,
// events flow, publishes leave under the §12.1 domain-separated signed-frame envelope with
// channel-scoped durable sequence numbers, and the run journals end-to-end through the REAL A1
// crash-safe segmented substrate (born audited — §8).
//
// Dev/test harness: shells `cargo build` for the guests and reads the `.wasm` (the same pattern
// as `worker_protocol.rs`), so the fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_abi::{DEFAULT_CHANNEL_CONTROL_ID, FRAME_ENVELOPE_DOMAIN_V2};
use daemon_vhc_host::run::{start_run, JournalSink, RunConfig, RunEnd, RunIdentity, SinkError};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_observe::journal::record::{
    ClockRec, DropId, DropRec, EventRec, ExecutionGrantRec, InitRec, InstantiationRec, RunHeader,
    SignedFrameRec, TerminalRec, TimerArmRec, TimerCancelRec, TrapInfo,
};
use daemon_vhc_observe::journal::{Body, ExecIdentity, Journal, RotatePolicy, StaticKey};
use daemon_vhc_proto::sign::verify_bytes;
use daemon_vhc_proto::{to_canonical_vec, Hash, PeerId, Signature};

// -- guest build harness (mirrors worker_protocol.rs) ----------------------------------------------

fn toy_averager_wasm() -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm("toy_averager")
}

// -- the JournalSink adapter over the REAL A1 substrate ---------------------------------------------

/// Adapts A1's `Journal` (crash-safe segments, commit barriers, sidecars) onto the driver's
/// dependency-inverted `JournalSink` seam — the wiring the session/worker will own in production.
struct JournalAdapter {
    journal: Journal<StaticKey>,
    id: ExecIdentity,
}

impl JournalSink for JournalAdapter {
    fn run_header(
        &mut self,
        abi: u64,
        worlds: &[(String, u64)],
        bridge: bool,
        manifest: &[u8],
        config: &[u8],
        grants: &[u8],
        resources: daemon_vhc_host::run::RunHeaderResources<'_>,
        channels: &[u8],
        device: &[u8],
    ) -> Result<(), SinkError> {
        // This adapter journals the legacy branch only; the composed branch is exercised through the
        // durable sink, which is the one a certification run writes with.
        let daemon_vhc_host::run::RunHeaderResources::Declared(claim) = resources else {
            panic!("the event-loop adapter is a lower-minor seat and records a declared claim");
        };
        self.journal
            .append(Body::RunHeader(Box::new(RunHeader {
                run_id: self.id.run_id,
                epoch: self.id.epoch,
                role: self.id.role.clone(),
                instance: self.id.instance,
                module: self.id.module,
                abi,
                worlds: worlds.iter().cloned().collect(),
                bridge,
                manifest: manifest.to_vec(),
                config: config.to_vec(),
                grants: grants.to_vec(),
                claim: Some(claim.to_vec()),
                channels: channels.to_vec(),
                device: device.to_vec(),
                resource_plan: None,
                resource_plan_hash: None,
                physical_claim: None,
                physical_claim_hash: None,
                aggregate_claim: None,
                aggregate_claim_hash: None,
                execution_grant: None,
                execution_grant_hash: None,
                format: u64::from(daemon_vhc_observe::journal::format_version()),
            })))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn instantiation(&mut self, counter: u64, reason: u64, at: u64) -> Result<(), SinkError> {
        self.journal.set_instantiation_counter(counter);
        self.journal
            .append(Body::Instantiation(InstantiationRec {
                counter,
                reason,
                at,
            }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn init(
        &mut self,
        config_hash: [u8; 32],
        grants_hash: [u8; 32],
        status: u64,
    ) -> Result<(), SinkError> {
        self.journal
            .append(Body::Init(InitRec {
                config_hash: Hash(config_hash),
                grants_hash: Hash(grants_hash),
                status,
            }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn execution_grant(&mut self, hash: [u8; 32], status: u64) -> Result<(), SinkError> {
        self.journal
            .append(Body::ExecutionGrant(ExecutionGrantRec {
                execution_grant_hash: Hash(hash),
                status,
            }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn event(&mut self, at: u64, frame: &[u8]) -> Result<(), SinkError> {
        self.journal
            .append(Body::Event(EventRec {
                at,
                frame: frame.to_vec(),
            }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn signed_frame(
        &mut self,
        channel: u64,
        seq: u64,
        sender: [u8; 32],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        self.journal
            .append(Body::SignedFrame(SignedFrameRec {
                channel,
                seq,
                sender: Hash(sender),
                frame: Some(frame.to_vec()),
                evidence: None,
            }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn next_seq(&mut self, channel: u64) -> u64 {
        self.journal.next_seq(channel)
    }

    fn publish(
        &mut self,
        channel: u64,
        seq: u64,
        payload: &[u8],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        // The A1 substrate allocates the durable seq itself under the same §8.4 barrier; the
        // driver derived its seq from `next_seq`, so the two must agree — asserted, not assumed.
        let (_, journal_seq) = self
            .journal
            .publish(channel, payload, frame.to_vec())
            .map_err(|e| SinkError(e.to_string()))?;
        assert_eq!(
            journal_seq, seq,
            "driver/journal seq agreement (§8.4 rule 2)"
        );
        Ok(())
    }

    fn clock(&mut self, now: u64) -> Result<(), SinkError> {
        self.journal
            .append(Body::Clock(ClockRec { now }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn timer_arm(&mut self, id: u64, delay: u64, armed_at: u64) -> Result<(), SinkError> {
        self.journal
            .append(Body::TimerArm(TimerArmRec {
                id,
                delay,
                armed_at,
            }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn timer_cancel(&mut self, id: u64, status: u64) -> Result<(), SinkError> {
        self.journal
            .append(Body::TimerCancel(TimerCancelRec { id, status }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn read_back(
        &mut self,
        src: u64,
        kind: u64,
        status: u64,
        value: &[u8],
    ) -> Result<(), SinkError> {
        // The substrate routes oversize values to encrypted sidecars itself (§8.5).
        self.journal
            .read_back(src, kind, status, value)
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn device_profile(&mut self, profile: &[u8]) -> Result<(), SinkError> {
        self.journal
            .append(Body::DeviceProfile(
                daemon_vhc_observe::journal::record::DeviceProfileRec {
                    profile: profile.to_vec(),
                },
            ))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn drop_coalesced(
        &mut self,
        class: u64,
        rule: u64,
        dropped: daemon_vhc_host::run::Dropped,
    ) -> Result<(), SinkError> {
        self.journal
            .append(Body::Drop(DropRec {
                class,
                rule,
                dropped: DropId {
                    hash: dropped.hash.map(Hash),
                    timer_id: dropped.timer_id,
                    channel: dropped.channel,
                    sender: dropped.sender.map(Hash),
                    seq: dropped.seq,
                },
            }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn condition(&mut self, code: &str, detail: &str) -> Result<(), SinkError> {
        self.journal
            .append(Body::Condition(
                daemon_vhc_observe::journal::record::ConditionRec {
                    code: code.to_string(),
                    detail: detail.to_string(),
                },
            ))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn completion(&mut self, op: u64, result: &[u8]) -> Result<(), SinkError> {
        self.journal
            .append(Body::Completion(
                daemon_vhc_observe::journal::record::CompletionRec {
                    op,
                    result: result.to_vec(),
                },
            ))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn snapshot(&mut self, manifest: &[u8]) -> Result<(), SinkError> {
        // tag 10, committed: §8.4 rule 2 — the barrier crosses before `snapshot_state` returns
        // `Accepted` to the guest (the upgrade transaction's durability point).
        self.journal
            .append_committed(Body::Snapshot(
                daemon_vhc_observe::journal::record::SnapshotRec {
                    manifest: manifest.to_vec(),
                },
            ))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }

    fn terminal(
        &mut self,
        kind: u64,
        outcome: Option<u64>,
        trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError> {
        self.journal
            .append_committed(Body::Terminal(TerminalRec {
                kind,
                outcome,
                trap: trap.map(|(code, import, context, detail)| TrapInfo {
                    code,
                    import,
                    context,
                    detail,
                }),
            }))
            .map(|_| ())
            .map_err(|e| SinkError(e.to_string()))
    }
}

// -- the end-to-end acceptance -----------------------------------------------------------------------

fn temp_root(tag: &str) -> PathBuf {
    let mut base = std::env::temp_dir();
    base.push(format!(
        "vhc-a2-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    base
}

/// The full loop: selection admits the major-2 guest → the driver runs it → timer events flow →
/// publishes leave signed + sequenced → `Stop` → Outcome `Ok` — journaled through the real A1
/// substrate, whose records this test then re-reads and verifies.
#[test]
fn toy_averager_runs_end_to_end_under_the_v2_driver() {
    let wasm = toy_averager_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");

    // ABI §1.3 selection: since the A2 flip, a well-formed major-2 module is ADMITTED.
    let sel = select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("major-2 module admitted to the event-loop driver");
    assert_eq!(sel.driver, daemon_vhc_abi::CandidateDriver::V2);
    // Minor 1 since B2: the toy consumes the sys@2 ambient surface (rng_seed/device_profile).
    assert_eq!((sel.major, sel.minor), (2, 1));

    // The real A1 journal, keyed by the frozen execution identity (§8.1).
    let identity = RunIdentity {
        run_id: [0xAA; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let exec_id = ExecIdentity {
        run_id: Hash(identity.run_id),
        epoch: identity.epoch,
        role: identity.role.clone(),
        instance: identity.instance,
        module: Hash(identity.module),
    };
    let root = temp_root("toy-averager");
    let journal = Journal::create(
        &root,
        exec_id,
        StaticKey::new([7u8; 32]),
        RotatePolicy::default(),
    )
    .expect("journal create");
    let adapter = Arc::new(Mutex::new(JournalAdapter {
        journal,
        id: ExecIdentity {
            run_id: Hash(identity.run_id),
            epoch: identity.epoch,
            role: identity.role.clone(),
            instance: identity.instance,
            module: Hash(identity.module),
        },
    }));

    // Config: average over 3 timer ticks. The device profile (8 GiB VRAM) feeds the guest's
    // module autotune (architecture §3.5); the identity feeds the deterministic rng_seed.
    let device = daemon_vhc_host::run::DeviceProfile {
        gpu: true,
        vram_bytes: 8 << 30,
        ram_bytes: 16 << 30,
        disk_bytes: 100 << 30,
    };
    let expected_seed = daemon_vhc_host::run::driver::derive_rng_seed(&identity);
    let mut run_cfg = RunConfig::new(identity, [0x51; 32], vec![3u8], b"grants-tbd".to_vec());
    run_cfg.device_bytes = device.to_wire();
    let run = start_run(&worker, &wasm, run_cfg, Box::new(adapter.clone())).expect("start run");

    // The guest arms its own timers (5 ms) and publishes after each — wait for the 3 publishes.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let published = run.pump.published();
        if published.len() >= 3 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for publishes; got {}",
            published.len()
        );
        std::thread::sleep(Duration::from_millis(5));
    }

    // Stop the run (RunComplete); the guest returns Outcome Ok.
    let pump = run.pump.clone();
    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    let end = run.wait().expect("guest thread clean");
    match end {
        RunEnd::Outcome(code) => assert_eq!(code, daemon_vhc_abi::OUTCOME_OK),
        other => panic!("expected Outcome(Ok), got {other:?}"),
    }

    // -- module autotune (architecture §3.5): the GUEST picked its batch size from the profile --
    // 8 GiB VRAM → the toy's ladder answers 8; the seed metric is the identity-derived
    // `derive_rng_seed` value — proving the module read the same profile the probe measures and
    // the same seed replay will re-derive.
    let metrics = pump.metrics();
    let metric = |name: &str| -> f64 {
        metrics
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("metric {name} missing: {metrics:?}"))
            .1
    };
    assert_eq!(
        metric("autotune.micro_batch"),
        8.0,
        "the module adapted its micro-batch to the journaled profile"
    );
    assert_eq!(
        metric("rng.seed0"),
        f64::from(u32::from_le_bytes([
            expected_seed[0],
            expected_seed[1],
            expected_seed[2],
            expected_seed[3]
        ])),
        "the guest observed the identity-derived deterministic seed"
    );

    // -- the publishes: §12.1 signed frames with channel-scoped durable seqs ----------------------
    let published = pump.published();
    assert!(published.len() >= 3);
    for (i, (channel, seq, frame)) in published.iter().take(3).enumerate() {
        assert_eq!(*channel, u64::from(DEFAULT_CHANNEL_CONTROL_ID));
        assert_eq!(*seq, i as u64, "durable seq is dense + monotone from 0");
        // [envelope, payload, sig] — verify like a third party (§12.1/§12.2).
        let v: Value = ciborium::de::from_reader(frame.as_slice()).expect("frame cbor");
        let Value::Array(parts) = v else {
            panic!("frame shape")
        };
        let Value::Map(env) = &parts[0] else {
            panic!("envelope shape")
        };
        let get = |k: &str| {
            env.iter()
                .find(|(key, _)| matches!(key, Value::Text(t) if t == k))
                .map(|(_, val)| val.clone())
                .unwrap_or_else(|| panic!("envelope field {k}"))
        };
        assert_eq!(get("domain"), Value::from(FRAME_ENVELOPE_DOMAIN_V2));
        assert_eq!(get("run_id"), Value::Bytes(vec![0xAA; 32]));
        assert_eq!(get("role"), Value::from("trainer"));
        assert_eq!(get("instance"), Value::from(1u64));
        assert_eq!(get("seq"), Value::from(i as u64));
        let Value::Bytes(payload) = &parts[1] else {
            panic!("payload")
        };
        assert_eq!(payload.len(), 8, "the averager publishes an f64 mean");
        assert_eq!(
            get("payload_hash"),
            Value::Bytes(blake3::hash(payload).as_bytes().to_vec())
        );
        let Value::Bytes(sender) = get("sender") else {
            panic!("sender")
        };
        let Value::Bytes(sig) = &parts[2] else {
            panic!("sig")
        };
        let env_bytes = to_canonical_vec(&parts[0]).expect("canonical envelope");
        verify_bytes(
            &PeerId(sender.as_slice().try_into().unwrap()),
            &Signature(sig.as_slice().try_into().unwrap()),
            &env_bytes,
        )
        .expect("frame signature verifies");
    }

    // -- the journal: born audited (§8) — re-read the REAL segments from disk ---------------------
    {
        let mut guard = adapter.lock().expect("adapter");
        guard.journal.commit().expect("final commit barrier");
        let records = guard.journal.read_all_records().expect("chain verifies");
        let tags: Vec<u8> = records.iter().map(|r| r.body.tag()).collect();

        // Header ordering: run-header first, then instantiation, then init (§8.3/§9.4).
        assert_eq!(tags[0], 0, "run header first");
        assert_eq!(tags[1], 13, "instantiation before any guest code");
        assert_eq!(tags[2], 11, "da_init journaled");

        // The delivered event sequence + observations all present.
        let count = |t: u8| tags.iter().filter(|&&x| x == t).count();
        assert!(count(5) >= 3, "timer arms journaled (tag 5): {tags:?}");
        assert!(
            count(1) >= 4,
            "delivered events journaled (tag 1): {tags:?}"
        );
        assert_eq!(count(4), 3, "exactly 3 publishes journaled (tag 4)");
        assert_eq!(
            count(15),
            1,
            "the device-profile delivery journaled once (tag 15): {tags:?}"
        );
        // The recorded profile is byte-identical to the admitted one (replay's input).
        let Some(Body::DeviceProfile(dp)) = records.iter().map(|r| &r.body).find(|b| b.tag() == 15)
        else {
            panic!("device-profile record present");
        };
        assert_eq!(
            dp.profile,
            device.to_wire(),
            "tag 15 records the delivered bytes"
        );
        assert_eq!(count(9), 1, "exactly one terminal fact (tag 9)");
        assert_eq!(tags.last(), Some(&9), "the terminal record is last");

        // Every journaled delivered-event frame decodes under the v2 codec (replay substrate).
        for r in &records {
            if let Body::Event(e) = &r.body {
                daemon_vhc_host::run::decode_event_frame(&e.frame)
                    .expect("journaled frame decodes");
            }
        }
        // The terminal is Outcome(0).
        let Some(Body::Terminal(t)) = records.iter().map(|r| &r.body).find(|b| b.tag() == 9) else {
            panic!("terminal record present");
        };
        assert_eq!((t.kind, t.outcome), (0, Some(0)), "Outcome Ok journaled");
    }

    let _ = std::fs::remove_dir_all(&root);
}

/// The channel table — not the guest — owns routing (§6.2): the same guest configured to publish
/// on an undeclared channel (config byte 1 = 9) traps a typed `GrantViolation` on its first
/// publish. The trap is a contained, journaled terminal fact (tag 9 kind 1) — the run instance
/// dies typed, the host survives (§7.6), and the journal names the offending import.
#[test]
fn publish_on_undeclared_channel_traps_grant_violation() {
    let wasm = toy_averager_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let identity = RunIdentity {
        run_id: [0xBB; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 2,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let exec_id = ExecIdentity {
        run_id: Hash(identity.run_id),
        epoch: 0,
        role: identity.role.clone(),
        instance: identity.instance,
        module: Hash(identity.module),
    };
    let root = temp_root("undeclared-channel");
    let journal = Journal::create(
        &root,
        exec_id.clone(),
        StaticKey::new([7u8; 32]),
        RotatePolicy::default(),
    )
    .expect("journal create");
    let adapter = Arc::new(Mutex::new(JournalAdapter {
        journal,
        id: exec_id,
    }));

    // Config: 1 tick, publish on undeclared channel 9.
    let run_cfg = RunConfig::new(identity, [0x52; 32], vec![1u8, 9u8], Vec::new());
    let run = start_run(&worker, &wasm, run_cfg, Box::new(adapter.clone())).expect("start run");

    let end = run.wait().expect("guest thread clean");
    match end {
        RunEnd::Trapped(trap) => {
            assert_eq!(trap.code, daemon_vhc_host::TrapCode::GrantViolation);
            assert!(
                trap.detail.contains("undeclared channel 9"),
                "{}",
                trap.detail
            );
        }
        other => panic!("expected a GrantViolation trap, got {other:?}"),
    }

    // The journal carries the typed terminal fact (kind 1 = trap) with the offending import.
    {
        let mut guard = adapter.lock().expect("adapter");
        guard.journal.commit().expect("commit");
        let records = guard.journal.read_all_records().expect("chain verifies");
        let Some(Body::Terminal(t)) = records.iter().map(|r| &r.body).find(|b| b.tag() == 9) else {
            panic!("terminal record present");
        };
        assert_eq!(t.kind, 1, "terminal kind = trap");
        let trap = t.trap.as_ref().expect("trap info");
        assert_eq!(trap.code, "GrantViolation");
        assert_eq!(trap.import, "publish");
        // And no publish record exists — the trap fired before any seq was allocated (§6.2).
        assert!(
            !records.iter().any(|r| r.body.tag() == 4),
            "no tag-4 publish for a refused channel"
        );
    }

    let _ = std::fs::remove_dir_all(&root);
}
