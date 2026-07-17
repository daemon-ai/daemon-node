// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The replay oracle — PROTO-20 as a library (spec §6.4 I1, §9, §11.2).
//!
//! Given a run's genesis-derived initial [`CoordinatorState`] + the recorded driving trace, re-run
//! the run through the **sandboxed coordinator module** and verify that every recorded
//! [`RoundRecord`] matches what the module re-derives — the "anyone can re-derive the coordinator"
//! property, and the substrate under resync (§9).
//!
//! ## Consensus runs only in the sandbox — even in verification
//!
//! Consensus is a wasm module, not a native host service (architecture §4.1). The replay oracle
//! therefore never re-runs consensus natively: it drives the same content-addressed
//! `coordinator-quorum` module the live run used, under the host runtime, through the
//! [`CoordinatorSandbox`] seam. This crate stays host-free (it is a node-side log tool below the
//! host runtime in the dependency graph), so it defines only the seam; the concrete driver — start
//! the pinned module, deliver recorded frames, collect its published decisions — lives in a
//! host-capable crate and is injected. The recorded driving trace carries only the coordinator's
//! *inputs* (signed worker messages); the module owns its own deterministic logical clock (one tick
//! per delivered frame), so recorded `Input::Clock`s are not re-fed — the sandbox re-derives them.
//!
//! The oracle consumes only signed messages + published records: the coordinator's own published
//! [`RoundRecord`]s carried in the trace are the **oracle** (compared, never delivered to the
//! module as an input); everything else is the driving trace fed to the module.

use daemon_vhc_proto::messages::{RoundRecord, SignedMessage, SwarmMessage};
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash, Seed};

use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, Input};

use crate::capture::RunCapture;
use crate::log::{MessageKind, MessageLog};
use crate::ObserveError;

/// Domain tag for the coordinator genesis seed derivation (see [`genesis_seed`]).
const GENESIS_SEED_DOMAIN: &[u8] = b"daemon-swarm/observe/genesis-seed/genesis/v2";

/// The deterministic genesis seed for a run, derived from its **genesis hash** (the run's
/// cryptographic identity, `FrozenGenesis::run_id`) by a domain-separated blake3.
///
/// Seed domains derive from the genesis hash, not from a frozen v1 envelope's canonical bytes: run
/// identity is anchored on the genesis, so the oracle reconstructs the exact `CoordinatorState::new`
/// a driver started from using only the public run identity, without any privileged input.
#[must_use]
pub fn genesis_seed(genesis_hash: &Hash) -> Seed {
    let mut buf = Vec::with_capacity(GENESIS_SEED_DOMAIN.len() + 32);
    buf.extend_from_slice(GENESIS_SEED_DOMAIN);
    buf.extend_from_slice(genesis_hash.as_bytes());
    Seed(*blake3_hash(&buf).as_bytes())
}

/// The sandbox seam the replay oracle drives consensus through (consensus never runs outside
/// the sandboxed, content-addressed coordinator module, even in verification).
///
/// A concrete implementation lives in a host-capable crate: it starts the run's pinned
/// `coordinator-quorum` module under the host runtime — configured from the genesis-derived
/// `initial` state (the opaque `da_init` config the module is initialized with) — delivers the
/// recorded driving `messages` as host-verified authoritative frames, waits for the module to
/// publish its decisions, and returns every published [`SwarmMessage`] in order. The module owns
/// its logical clock, so no clocks are delivered.
pub trait CoordinatorSandbox {
    /// Re-derive a recorded run inside the sandbox: start the pinned coordinator module from
    /// `initial`, deliver `messages` in order, and return every decision the module published, in
    /// order. `expected_records` is how many [`RoundRecord`]s the recorded run published (the oracle
    /// count), so the driver can wait deterministically for the module to finish before stopping it.
    ///
    /// # Errors
    /// [`ReplayError::Sandbox`] if the module fails to start, a frame is refused, or the module does
    /// not produce the expected decisions before its deadline.
    fn replay_run(
        &self,
        initial: &CoordinatorState,
        messages: &[SignedMessage],
        expected_records: usize,
    ) -> Result<Vec<SwarmMessage>, ReplayError>;
}

/// A successful replay: what the sandboxed coordinator module re-derived.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayReport {
    /// The re-derived round records, in production order.
    pub records: Vec<RoundRecord>,
    /// How many recorded `RoundRecord`s in the trace were checked (and matched) against re-derivation.
    pub rounds_verified: u64,
    /// blake3 of the canonical CBOR of the re-derived decision stream (the resync anchor, I1).
    ///
    /// The observer consumes only published objects, never the module's privileged internal state,
    /// so the anchor is a pure function of the module's published `RoundRecord`s: two runs that
    /// re-derive the same decision stream carry the same anchor.
    pub final_state_hash: Hash,
}

/// The first divergence between a recorded record and the re-derived one (§6.4 I1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayDivergence {
    /// The round at which re-derivation and the record disagree.
    pub round: u64,
    /// The recorded (oracle) round record.
    pub recorded: RoundRecord,
    /// What the module re-derived for that round (`None` if it produced no record).
    pub rederived: Option<RoundRecord>,
    /// A human-readable summary of the mismatch.
    pub detail: String,
}

/// Why a replay did not complete: a setup failure, a sandbox failure, or a pinpointed first divergence.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// The genesis config could not be resolved (not a divergence).
    #[error("replay setup: {0}")]
    Setup(#[from] ObserveError),
    /// The sandboxed coordinator module could not be driven (start / delivery / deadline).
    #[error("coordinator sandbox: {0}")]
    Sandbox(String),
    /// A recorded record diverged from the re-derivation (the PROTO-20 failure). Boxed to keep the
    /// error small (the divergence carries two full round records).
    #[error("replay diverged at round {}: {}", .0.round, .0.detail)]
    Diverged(Box<ReplayDivergence>),
}

/// Split a recorded `tick` trace into the driving messages (fed to the module) and the oracle
/// [`RoundRecord`]s (compared, never fed). The module owns its clock, so `Input::Clock`s are
/// dropped; `RoundOpen` is a module output and is dropped too.
fn partition_trace(inputs: impl Iterator<Item = Input>) -> (Vec<SignedMessage>, Vec<RoundRecord>) {
    let mut driving = Vec::new();
    let mut oracle = Vec::new();
    for input in inputs {
        if let Input::Message(sm) = input {
            match &sm.payload {
                SwarmMessage::RoundRecord(r) => oracle.push(r.clone()),
                SwarmMessage::RoundOpen(_) => {}
                _ => driving.push(sm),
            }
        }
        // Clocks and control are not driving frames here: the module owns its logical clock, and a
        // signed control request rides its own channel (not exercised by the recorded traces).
    }
    (driving, oracle)
}

/// Re-derive a run inside the sandbox from a **given initial [`CoordinatorState`]** and verify the
/// recorded records match (§6.4 I1, PROTO-20).
///
/// The initial state is the genesis-derived config the coordinator module is started with (the
/// in-process harness / gate ceremony builds it directly and captures it in a
/// [`RunCapture`](crate::capture::RunCapture)). `inputs` carries the driving trace (signed worker
/// messages) followed by the wire-recorded `RoundRecord`s as the oracle. A `RoundRecord` in the
/// stream is **compared** against what the module re-derived; `RoundOpen`/clocks are not delivered.
///
/// # Errors
///
/// [`ReplayError::Diverged`] on the first recorded record that disagrees with the re-derivation;
/// [`ReplayError::Sandbox`] on a module start/delivery/deadline failure.
pub fn replay_from_state(
    sandbox: &dyn CoordinatorSandbox,
    initial: CoordinatorState,
    inputs: impl Iterator<Item = Input>,
) -> Result<ReplayReport, ReplayError> {
    let (driving, oracle) = partition_trace(inputs);
    let published = sandbox.replay_run(&initial, &driving, oracle.len())?;

    // The module's published RoundRecords, in production order — the re-derivation.
    let records: Vec<RoundRecord> = published
        .into_iter()
        .filter_map(|m| match m {
            SwarmMessage::RoundRecord(r) => Some(r),
            _ => None,
        })
        .collect();

    // Compare each oracle record against the re-derived record for the same round, in order.
    let mut rounds_verified = 0u64;
    for recorded in &oracle {
        let rederived = records.iter().find(|r| r.round == recorded.round);
        match rederived {
            Some(r) if r == recorded => rounds_verified += 1,
            Some(r) => {
                return Err(ReplayError::Diverged(Box::new(ReplayDivergence {
                    round: recorded.round,
                    recorded: recorded.clone(),
                    rederived: Some(r.clone()),
                    detail: "recorded RoundRecord differs from the re-derived record".into(),
                })));
            }
            None => {
                return Err(ReplayError::Diverged(Box::new(ReplayDivergence {
                    round: recorded.round,
                    recorded: recorded.clone(),
                    rederived: None,
                    detail: "re-derivation produced no record for this round".into(),
                })));
            }
        }
    }

    let final_state_hash = decision_stream_anchor(&records)?;
    Ok(ReplayReport {
        records,
        rounds_verified,
        final_state_hash,
    })
}

/// The resync anchor (I1): blake3 of the canonical CBOR of the module's published decision stream.
fn decision_stream_anchor(records: &[RoundRecord]) -> Result<Hash, ReplayError> {
    let bytes = to_canonical_vec(records).map_err(|e| ObserveError::Codec(e.to_string()))?;
    Ok(blake3_hash(&bytes))
}

/// Verify a recorded run: re-derive it inside the sandbox from the [`RunCapture`]'s initial state
/// over its driving trace, using the **independent** wire [`MessageLog`]'s `RoundRecord`s as the
/// oracle (§6.4 I1).
///
/// The capture supplies the driving inputs (signed worker messages) the coordinator consumed; the
/// log supplies what the coordinator actually broadcast. The module re-derives a `RoundRecord` per
/// round; each logged record is compared against it. A successful [`ReplayReport`] with
/// `rounds_verified` equal to the logged record count proves the run's per-round consensus
/// (committed set + drops = the round digest) is byte-reproducible — the `swarm-replay`
/// gate-ceremony assertion.
///
/// # Errors
///
/// [`ReplayError::Diverged`] at the first logged record that does not re-derive;
/// [`ReplayError::Sandbox`] on a module failure.
pub fn replay_capture(
    sandbox: &dyn CoordinatorSandbox,
    capture: RunCapture,
    oracle: &MessageLog,
) -> Result<ReplayReport, ReplayError> {
    let oracle_records: Vec<Input> = oracle
        .by_kind(MessageKind::RoundRecord)
        .cloned()
        .map(Input::Message)
        .collect();
    let inputs = capture.inputs.into_iter().chain(oracle_records);
    replay_from_state(sandbox, capture.initial, inputs)
}

/// How many `RoundRecord`s a [`MessageLog`] carries (the count `replay_capture` verifies against).
#[must_use]
pub fn logged_round_records(log: &MessageLog) -> usize {
    log.by_kind(MessageKind::RoundRecord).count()
}
