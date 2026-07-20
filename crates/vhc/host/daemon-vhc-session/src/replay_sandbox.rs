// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator replay sandbox — the concrete [`CoordinatorSandbox`] the replay oracle
//! drives consensus through (spec §6.4 I1; architecture §4.1).
//!
//! **HARNESS-ERA** (`#[cfg(any(test, feature = "harness"))]`): needs the host runtime + observe
//! (oracle tooling), so it rides the `harness` seat; production sessions never re-derive
//! consensus locally.
//!
//! Consensus is a wasm module, not a native host service, so the replay oracle re-derives a recorded
//! run inside the **same content-addressed `coordinator-quorum` module** the live run used, under the
//! real major-2 event-loop driver — never a native `tick` (consensus never runs outside the
//! sandbox, even in verification). `daemon-vhc-observe` sits below the host runtime in the
//! dependency graph and only defines the [`CoordinatorSandbox`] seam; this module — in the session
//! crate, which links the host runtime — is the driver that fulfils it.
//!
//! ## Config source and module selection
//!
//! The oracle's initial [`CoordinatorState`] is the run's genesis-derived coordinator config (the
//! opaque `da_init` bytes the module is initialized with — architecture §5.1 seam rule): the
//! sandbox encodes it as the guest's `{ "state": <CoordinatorState> }` init, selects the driver for
//! the coordinator blob (which must be major-2), and `start_run`s it. The recorded, already-signed
//! worker messages are delivered as host-verified authoritative frames on the records channel; the
//! module owns its own deterministic logical clock (one synthetic tick per delivered frame), so no
//! clocks are delivered — the module re-derives them.
//!
//! ## Determinism
//!
//! The module is content-addressed (blake3 of the wasm), its `tick` body is deterministic, and it is
//! driven over the identical ordered frames the run recorded, so its published decision stream is a
//! pure function of `(module, initial state, frames)` — a recorded run and its replay produce
//! byte-identical `RoundRecord`s. The dual-compilation identity gate pins that this module's wasm32
//! decisions equal the native `tick` reference over identical inputs.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use ciborium::value::Value;

use daemon_vhc_abi::{CandidateDriver, STOP_REASON_RUN_COMPLETE};
use daemon_vhc_host::run::{start_run, DeliverVerdict, MemorySink, RunConfig, RunIdentity};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_observe::{CoordinatorSandbox, ReplayError};
use daemon_vhc_proto::{from_canonical_slice, to_canonical_vec};
use daemon_vhc_sdk_consensus::coordinator::CoordinatorState;
use daemon_vhc_sdk_consensus::{SignedMessage, VhcMessage};

/// Frame delivery / drain deadline: a re-derivation that has not drained by here is a sandbox fault.
const RUN_DEADLINE: Duration = Duration::from_secs(60);
/// Quiet window with no new publishes that marks the module's decision stream complete.
const DRAIN_QUIET: Duration = Duration::from_millis(300);
/// Fixed key seed for the coordinator's frame signer: the replay oracle compares the module's
/// published `RoundRecord` **payloads**, never their transport signatures, so any seed serves.
/// The registry constant.
use daemon_vhc_proto::domains::REPLAY_SANDBOX_FRAME_KEY_SEED as FRAME_KEY_SEED;

/// A [`CoordinatorSandbox`] backed by the production `coordinator-quorum` wasm module running under
/// the real major-2 host driver.
pub struct SandboxedCoordinator {
    wasm: Vec<u8>,
}

impl SandboxedCoordinator {
    /// A sandbox over an explicit coordinator wasm blob (the run's genesis-pinned coordinator
    /// module bytes).
    #[must_use]
    pub fn new(coordinator_wasm: Vec<u8>) -> Self {
        Self {
            wasm: coordinator_wasm,
        }
    }

    /// A sandbox over the `coordinator-quorum` guest built from source — the dev / gate-ceremony
    /// path (the established testkit pattern: shell `cargo build` for the guests workspace).
    ///
    /// # Errors
    /// [`ReplayError::Sandbox`] if the guest build fails or the artifact cannot be read.
    pub fn from_built_guest() -> Result<Self, ReplayError> {
        Ok(Self::new(coordinator_quorum_wasm()?))
    }
}

impl CoordinatorSandbox for SandboxedCoordinator {
    fn replay_run(
        &self,
        initial: &CoordinatorState,
        messages: &[SignedMessage],
        expected_records: usize,
    ) -> Result<Vec<VhcMessage>, ReplayError> {
        let sandbox = |detail: String| ReplayError::Sandbox(detail);

        let module_hash = *blake3::hash(&self.wasm).as_bytes();
        let engine =
            Worker::new(EngineConfig::default()).map_err(|e| sandbox(format!("engine: {e}")))?;
        let sel = select_driver(&engine, &self.wasm, Some(&module_hash))
            .map_err(|e| sandbox(format!("coordinator selection: {e}")))?;
        if sel.driver != CandidateDriver::V2 {
            return Err(sandbox(format!(
                "coordinator module must select the major-2 driver, got {:?}",
                sel.driver
            )));
        }

        let identity = RunIdentity {
            run_id: initial.config.envelope_hash.0,
            epoch: 0,
            role: "coordinator".to_string(),
            instance: 0,
            module: module_hash,
        };
        let key_seed = *blake3::hash(FRAME_KEY_SEED).as_bytes();
        let config = guest_config(initial)?;
        let sink = std::sync::Arc::new(std::sync::Mutex::new(MemorySink::new()));
        let run_cfg = RunConfig::new(identity, key_seed, config, Vec::new());
        let run = start_run(&engine, &self.wasm, run_cfg, Box::new(sink))
            .map_err(|e| sandbox(format!("coordinator start_run: {e}")))?;
        let pump = run.pump.clone();

        // Deliver every recorded frame as a host-verified authoritative frame (records channel 0),
        // one dense seq per sender, back-pressuring on a full spool (never dropping a consensus
        // input). The module owns its clock, so no clocks are delivered.
        let mut seqs: BTreeMap<[u8; 32], u64> = BTreeMap::new();
        let deadline = Instant::now() + RUN_DEADLINE;
        for sm in messages {
            sm.verify()
                .map_err(|e| sandbox(format!("recorded frame REFUSED above the pump: {e}")))?;
            let sender = sm.signer.0;
            let payload = to_canonical_vec(&sm.payload)
                .map_err(|e| sandbox(format!("payload encode: {e}")))?;
            let evidence =
                to_canonical_vec(sm).map_err(|e| sandbox(format!("evidence encode: {e}")))?;
            let seq = seqs.entry(sender).or_insert(0);
            loop {
                match pump
                    .deliver_frame(0, *seq, sender, payload.clone(), evidence.clone())
                    .map_err(|e| sandbox(format!("deliver: {e}")))?
                {
                    DeliverVerdict::Accepted => break,
                    DeliverVerdict::SpoolFull | DeliverVerdict::SenderQuota => {
                        if Instant::now() >= deadline {
                            return Err(sandbox(
                                "coordinator spool never drained (back-pressure)".into(),
                            ));
                        }
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    other => return Err(sandbox(format!("unexpected deliver verdict: {other:?}"))),
                }
            }
            *seq += 1;
        }

        // Wait for the module to drain the frames into its decision stream: the published set is a
        // deterministic function of the inputs, so wait until it stops growing (a quiet window) and
        // carries at least the expected number of RoundRecords.
        let mut last_len = pump.published().len();
        let mut last_change = Instant::now();
        loop {
            let published = pump.published();
            if published.len() != last_len {
                last_len = published.len();
                last_change = Instant::now();
            }
            let records = count_records(&published);
            let quiet = last_change.elapsed() >= DRAIN_QUIET;
            if quiet && records >= expected_records {
                break;
            }
            if Instant::now() >= deadline {
                return Err(sandbox(format!(
                    "coordinator produced {records} records ({} publishes) before the deadline, \
                     expected at least {expected_records}",
                    published.len()
                )));
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        let published = pump.published();
        pump.stop(STOP_REASON_RUN_COMPLETE)
            .map_err(|e| sandbox(format!("coordinator stop: {e}")))?;
        run.wait()
            .map_err(|e| sandbox(format!("coordinator guest thread: {e}")))?;

        decode_published(&published).map_err(sandbox)
    }
}

/// Encode the guest's `da_init` config: `{ "state": <CoordinatorState> }` (the guest defaults
/// `tick_period_ms`/`control_channel` to 0 — the deterministic event-driven clock, channel 0).
fn guest_config(state: &CoordinatorState) -> Result<Vec<u8>, ReplayError> {
    let state_val = Value::serialized(state)
        .map_err(|e| ReplayError::Sandbox(format!("state to cbor value: {e}")))?;
    let init = Value::Map(vec![(Value::Text("state".into()), state_val)]);
    to_canonical_vec(&init).map_err(|e| ReplayError::Sandbox(format!("init cbor: {e}")))
}

/// Count published `RoundRecord`s in a raw `(channel, seq, frame)` publish stream.
fn count_records(published: &[(u64, u64, Vec<u8>)]) -> usize {
    published
        .iter()
        .filter_map(|(_, _, frame)| decode_frame(frame))
        .filter(|m| matches!(m, VhcMessage::RoundRecord(_)))
        .count()
}

/// Decode a publish stream (`(channel, seq, signed-frame bytes)`) into ordered messages.
fn decode_published(published: &[(u64, u64, Vec<u8>)]) -> Result<Vec<VhcMessage>, String> {
    published
        .iter()
        .map(|(_, _, frame)| decode_frame(frame).ok_or_else(|| "undecodable publish frame".into()))
        .collect()
}

/// Decode one `[envelope, payload, sig]` signed frame's payload into a [`VhcMessage`].
fn decode_frame(frame: &[u8]) -> Option<VhcMessage> {
    let v: Value = ciborium::de::from_reader(frame).ok()?;
    let Value::Array(parts) = v else { return None };
    let Value::Bytes(payload) = parts.get(1)? else {
        return None;
    };
    from_canonical_slice::<VhcMessage>(payload).ok()
}

/// Build (once) and read the production `coordinator-quorum.wasm` from the guests workspace via
/// the shared builder (`daemon-vhc-guest-build`: the reproducibility remap env + the cross-process
/// build lock live there, once). Shared with the in-process whole-run harness's coordinator
/// recording drive (`harness`), so a recorded run and its replay oracle drive the byte-identical
/// content-addressed module.
pub(crate) fn coordinator_quorum_wasm() -> Result<Vec<u8>, ReplayError> {
    daemon_vhc_guest_build::module_bytes("coordinator_quorum").map_err(ReplayError::Sandbox)
}
