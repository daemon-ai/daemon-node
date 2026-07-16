// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The reusable "native coordinator" half of a whole run (refactor §6): the pure
//! `daemon-vhc-coordinator` `tick` state machine wrapped with envelope config, clock advancement,
//! signing, and an outbox — so a whole run can be driven by a real coordinator in-process, ahead of
//! the D2 wasm coordinator. This is the coordinator building block the tiny-llama-v2 barrier-round
//! whole run composes with [`crate::run`] (the A2 t2 shape, lifted out of `daemon-vhc-worker`).
//!
//! `tick` emits UNSIGNED messages (the `LocalCoordinator` shell contract); [`NativeCoordinator`]
//! signs them with the run's coordinator key so a worker verifies them exactly as a remote peer
//! would (the §12 evidentiary path).

use std::collections::VecDeque;

use daemon_vhc_proto::messages::{Join, ThroughputClass};
use daemon_vhc_proto::{
    CapabilitySet, Envelope, Hash, IrohId, Seed, SignedMessage, SigningKey, SwarmMessage,
    SWARM_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{
    tick, CoordinatorParams, CoordinatorState, Input, Output, RunConfig,
};

/// A native coordinator driven in-process from an envelope.
pub struct NativeCoordinator {
    state: CoordinatorState,
    key: SigningKey,
    now_s: u64,
    outbox: VecDeque<SwarmMessage>,
    envelope_hash: Hash,
}

impl NativeCoordinator {
    /// Build from the run's envelope + the coordinator signing key. `seq_len` is the corpus
    /// sequence length the coordinator schedules with (the launch-DiLoCo `CoordinatorParams`
    /// otherwise take their zero defaults — witness/overlap/verification off, `k_absences = 8`).
    ///
    /// # Errors
    /// If `RunConfig::from_envelope` rejects the envelope.
    pub fn from_envelope(
        envelope: &Envelope,
        key: SigningKey,
        seq_len: u64,
    ) -> Result<Self, String> {
        let params = CoordinatorParams {
            seq_len,
            witness_target: 0,
            overlap_bps: 0,
            k_absences: 8,
            verification_percent: 0,
            authorized: Vec::new(),
        };
        let config = RunConfig::from_envelope(envelope, params)
            .map_err(|e| format!("coordinator config: {e}"))?;
        let envelope_hash = config.envelope_hash;
        let state = CoordinatorState::new(config, Seed([0x33; 32]), 0);
        Ok(Self {
            state,
            key,
            now_s: 0,
            outbox: VecDeque::new(),
            envelope_hash,
        })
    }

    /// The run's envelope hash (a joining worker binds its `Join` to it).
    #[must_use]
    pub fn envelope_hash(&self) -> Hash {
        self.envelope_hash
    }

    /// The coordinator's current logical clock (seconds).
    #[must_use]
    pub fn now_s(&self) -> u64 {
        self.now_s
    }

    /// The coordinator's sender identity bytes (its `PeerId`) — what a worker pump sees as the
    /// authoritative frame sender.
    #[must_use]
    pub fn sender(&self) -> [u8; 32] {
        daemon_vhc_proto::peer_id(&self.key).0
    }

    /// Advance the clock one second at a time until the coordinator produces a message, bounded
    /// by `max_ticks` — the timeout-driven drive (warmup, round cadence) with a typed failure
    /// instead of a hang when the coordinator goes quiet.
    ///
    /// # Errors
    /// On a coordinator rejection, or if `max_ticks` elapse without any output.
    pub fn advance_bounded(&mut self, max_ticks: u32) -> Result<(), String> {
        for _ in 0..max_ticks {
            self.advance_clock(1)?;
            if !self.outbox.is_empty() {
                return Ok(());
            }
        }
        Err("coordinator went quiet (advance_bounded exhausted)".into())
    }

    fn feed(&mut self, input: Input) -> Result<(), String> {
        let (next, outputs) = tick(self.state.clone(), input);
        self.state = next;
        for o in outputs {
            match o {
                Output::Publish(msg) => self.outbox.push_back(*msg),
                Output::Reject(r) => return Err(format!("coordinator rejected input: {r:?}")),
                Output::Note(_) => {}
            }
        }
        Ok(())
    }

    /// Admit a worker into the roster: feed a signed `Join` bound to this run's envelope hash.
    ///
    /// # Errors
    /// On sign failure or a coordinator rejection.
    pub fn join(&mut self, worker_key: &SigningKey, run_id: &str) -> Result<(), String> {
        let join = SwarmMessage::Join(Join {
            run_id: run_id.to_string(),
            iroh_id: IrohId([0x44; 32]),
            class: ThroughputClass::C1,
            capabilities: CapabilitySet::new(),
            envelope_hash: Some(self.envelope_hash),
        });
        let signed = SignedMessage::sign(worker_key, SWARM_PROTO_VERSION, join)
            .map_err(|e| format!("join sign: {e}"))?;
        self.feed(Input::Message(signed))
    }

    /// Advance the coordinator clock by `secs` and drive the timeout-based transitions.
    ///
    /// # Errors
    /// On a coordinator rejection.
    pub fn advance_clock(&mut self, secs: u64) -> Result<(), String> {
        self.now_s += secs;
        self.feed(Input::Clock(self.now_s))
    }

    /// Feed an inbound signed worker message (a `Commitment`, `StorageReceipt`, …).
    ///
    /// # Errors
    /// On a coordinator rejection.
    pub fn feed_message(&mut self, signed: SignedMessage) -> Result<(), String> {
        self.feed(Input::Message(signed))
    }

    /// Pop the next authoritative message the coordinator produced (`RoundOpen`, `RoundRecord`, …),
    /// unsigned — sign it with [`NativeCoordinator::sign`] before delivering it to a worker.
    pub fn next_message(&mut self) -> Option<SwarmMessage> {
        self.outbox.pop_front()
    }

    /// Sign a coordinator message with the run's coordinator key (the §12 evidentiary envelope a
    /// worker verifies above its pump).
    ///
    /// # Errors
    /// On sign failure.
    pub fn sign(&self, msg: SwarmMessage) -> Result<SignedMessage, String> {
        SignedMessage::sign(&self.key, SWARM_PROTO_VERSION, msg)
            .map_err(|e| format!("coordinator sign: {e}"))
    }
}
