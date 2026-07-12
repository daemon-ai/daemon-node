// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The replay oracle — PROTO-20 as a library (spec §6.4 I1, §9, §11.2).
//!
//! Given the run envelope + the recorded `tick` input trace, re-run
//! [`daemon_swarm_coordinator::tick`] from genesis and verify that every recorded
//! [`RoundRecord`] matches what the pure coordinator function re-derives — the "anyone can re-derive
//! the coordinator" property, and the substrate under resync (§9). The oracle consumes only signed
//! messages + published records: the coordinator's own published `RoundRecord`s carried in the input
//! trace are the **oracle** (compared, not fed back to `tick`); everything else drives `tick`.
//!
//! On the event-driven happy path a round finalizes with **zero clocks** (the Merge-2 P0 finding), so
//! the trace is pure messages; timeout/straggler rounds additionally carry the recorded
//! `Input::Clock`s (the driver's sidecar — clocks are not signed messages, §14).

use daemon_swarm_proto::envelope::Envelope;
use daemon_swarm_proto::messages::{RoundRecord, SwarmMessage};
use daemon_swarm_proto::{blake3_hash, to_canonical_vec, Hash, Seed};

use daemon_swarm_coordinator::{
    tick, CoordinatorParams, CoordinatorState, Input, Output, RunConfig,
};

use crate::ObserveError;

/// The deterministic genesis seed for a run, derived from its envelope (domain-separated blake3), so
/// the oracle reconstructs the exact `CoordinatorState::new` a driver started from without any
/// privileged input.
pub fn genesis_seed(env: &Envelope) -> Result<Seed, ObserveError> {
    let bytes = to_canonical_vec(env).map_err(|e| ObserveError::Codec(e.to_string()))?;
    let mut buf = Vec::with_capacity(bytes.len() + 32);
    buf.extend_from_slice(b"daemon-swarm/observe/genesis-seed/v1");
    buf.extend_from_slice(&bytes);
    Ok(Seed(*blake3_hash(&buf).as_bytes()))
}

/// A successful replay: what the pure coordinator re-derived.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayReport {
    /// The re-derived round records, in production order.
    pub records: Vec<RoundRecord>,
    /// How many recorded `RoundRecord`s in the trace were checked (and matched) against re-derivation.
    pub rounds_verified: u64,
    /// blake3 of the canonical CBOR of the final coordinator state (the resync anchor, I1).
    pub final_state_hash: Hash,
}

/// The first divergence between a recorded record and the re-derived one (§6.4 I1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayDivergence {
    /// The round at which re-derivation and the record disagree.
    pub round: u64,
    /// The recorded (oracle) round record.
    pub recorded: RoundRecord,
    /// What `tick` re-derived for that round (`None` if the coordinator produced no record).
    pub rederived: Option<RoundRecord>,
    /// A human-readable summary of the mismatch.
    pub detail: String,
}

/// Why a replay did not complete: a setup failure, or a pinpointed first divergence.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// The envelope config / genesis could not be resolved (not a divergence).
    #[error("replay setup: {0}")]
    Setup(#[from] ObserveError),
    /// A recorded record diverged from the re-derivation (the PROTO-20 failure). Boxed to keep the
    /// error small (the divergence carries two full round records).
    #[error("replay diverged at round {}: {}", .0.round, .0.detail)]
    Diverged(Box<ReplayDivergence>),
}

/// Re-run `tick` from genesis over `inputs` and verify recorded records match (§6.4 I1, PROTO-20).
///
/// `inputs` is the recorded `tick` trace: driving inputs (peer messages, storage receipts, clocks,
/// control) **plus** the coordinator's own signed `RoundRecord` publications, which serve as the
/// oracle. `RoundOpen`/`RoundRecord` messages are never fed back to `tick` (they are outputs); a
/// `RoundRecord` is compared against the record `tick` last produced for its round.
pub fn replay(
    env: &Envelope,
    params: CoordinatorParams,
    inputs: impl Iterator<Item = Input>,
) -> Result<ReplayReport, ReplayError> {
    let config =
        RunConfig::from_envelope(env, params).map_err(|e| ObserveError::Replay(e.to_string()))?;
    let seed = genesis_seed(env)?;
    let mut state = CoordinatorState::new(config, seed, 0);

    // The last record `tick` produced for each round (records[idx_by_round]).
    let mut produced_by_round: std::collections::BTreeMap<u64, RoundRecord> =
        std::collections::BTreeMap::new();
    let mut records: Vec<RoundRecord> = Vec::new();
    let mut rounds_verified = 0u64;

    for input in inputs {
        if let Input::Message(sm) = &input {
            match &sm.payload {
                // Oracle: compare, do not feed a coordinator output back into `tick`.
                SwarmMessage::RoundRecord(recorded) => {
                    let round = recorded.round;
                    match produced_by_round.get(&round) {
                        Some(rederived) if rederived == recorded => {
                            rounds_verified += 1;
                        }
                        Some(rederived) => {
                            return Err(ReplayError::Diverged(Box::new(ReplayDivergence {
                                round,
                                recorded: recorded.clone(),
                                rederived: Some(rederived.clone()),
                                detail: "recorded RoundRecord differs from the re-derived record"
                                    .into(),
                            })));
                        }
                        None => {
                            return Err(ReplayError::Diverged(Box::new(ReplayDivergence {
                                round,
                                recorded: recorded.clone(),
                                rederived: None,
                                detail: "re-derivation produced no record for this round".into(),
                            })));
                        }
                    }
                    continue;
                }
                // The coordinator's other output is also not a `tick` input.
                SwarmMessage::RoundOpen(_) => continue,
                _ => {}
            }
        }

        let (next, outputs) = tick(state, input);
        state = next;
        for out in outputs {
            if let Output::Publish(msg) = out {
                if let SwarmMessage::RoundRecord(r) = *msg {
                    produced_by_round.insert(r.round, r.clone());
                    records.push(r);
                }
            }
        }
    }

    let final_state_hash =
        blake3_hash(&to_canonical_vec(&state).map_err(|e| ObserveError::Codec(e.to_string()))?);
    Ok(ReplayReport {
        records,
        rounds_verified,
        final_state_hash,
    })
}
