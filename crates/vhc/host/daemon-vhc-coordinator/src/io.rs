// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `tick` inputs and outputs (spec §6.2, §6.4, §11.1).
//!
//! `tick(state, input) -> (state', outputs)` is I/O-free: **time enters as [`Input::Clock`]**, signed
//! messages as [`Input::Message`], and operator intents as [`Input::Control`]. The coordinator emits
//! its own messages **unsigned** ([`Output::Publish`]) — the Wave-3 harness signs + broadcasts them,
//! keeping `tick` key-free while the round stays provable from signed evidence (I6). Every input class
//! yields at least one output class (PROTO-1).

use daemon_swarm_proto::capability::Capability;
use daemon_swarm_proto::messages::{SignedMessage, SwarmMessage};
use daemon_swarm_proto::sign::Signed;
use daemon_swarm_proto::{PeerId, SwarmProtoVersion};
use serde::{Deserialize, Serialize};

use crate::state::Phase;

/// An operator control intent (pause/resume), signed by the requesting principal (§11.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlRequest {
    /// The run this intent targets.
    pub run_id: String,
    /// The requested action.
    pub action: ControlAction,
}

/// A pause/resume action (§6.2, §11.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlAction {
    /// Pause the run (authorized principals only).
    Pause,
    /// Resume a paused run (authorized principals only).
    Resume,
}

/// A single `tick` input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Input {
    /// A clock advance (unix seconds) — the only way time enters `tick`.
    Clock(u64),
    /// An inbound, already-signed swarm message.
    Message(SignedMessage),
    /// A signed operator control request (pause/resume).
    Control(Signed<ControlRequest>),
}

/// A non-wire state-change signal surfaced to the harness / observer (§14).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Notice {
    /// The lifecycle phase changed.
    PhaseChanged {
        /// Previous phase.
        from: Phase,
        /// New phase.
        to: Phase,
    },
    /// A join was admitted to the roster (or staged as pending).
    Admitted(PeerId),
    /// A peer was dropped after K record-absences (§6.4).
    Dropped(PeerId),
    /// Two peers reported divergent digests for a round (§6.4 desync detection).
    DigestMismatch {
        /// The round.
        round: u64,
        /// The two disagreeing peers.
        peers: (PeerId, PeerId),
    },
    /// The run reached `[data].stop` and finished (§6.2).
    Finished,
}

/// Why an input was rejected (typed).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rejection {
    /// The run is in a halted phase (`Uninitialized`/`Paused`/`Finished`; PROTO-14).
    Halted(Phase),
    /// A signature failed to verify.
    BadSignature,
    /// The message's proto version does not match the run (§16; PROTO-13).
    VersionMismatch {
        /// The run's pinned version.
        expected: SwarmProtoVersion,
        /// The message's version.
        got: SwarmProtoVersion,
    },
    /// A pause/resume from a non-authorized principal (§11.1; PROTO-14).
    Unauthorized,
    /// A join was declined at admission (§6.5).
    Admission(AdmissionReject),
    /// A coordinator-only message arrived inbound, or a message is invalid in this phase.
    UnexpectedMessage,
    /// A round message came from a peer not on the roster.
    UnknownPeer,
    /// An attestation came from a peer outside the round's witness set (§6.3).
    NotWitness,
    /// Round evidence for a round the ring no longer holds / has not opened.
    StaleRound {
        /// The current round.
        current: u64,
        /// The message's round.
        got: u64,
    },
    /// A control request targeting a different run.
    RunIdMismatch,
}

/// Why a join was declined at admission (§6.5; TDD PROTO-12/13).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionReject {
    /// Proto-version mismatch (exact match required, §16).
    VersionMismatch {
        /// The run's pinned version.
        expected: SwarmProtoVersion,
        /// The join's version.
        got: SwarmProtoVersion,
    },
    /// The asserted envelope hash does not match the run's frozen envelope (§6.1/§6.5).
    EnvelopeHashMismatch,
    /// The advertised capability set is missing required ops (`required ⊄ advertised`, §6.5).
    MissingCapabilities(Vec<Capability>),
    /// The roster (incl. pending) is at `max_peers`.
    RosterFull,
    /// Joins are not accepted in this phase (halted).
    NotAccepting(Phase),
    /// The peer is already an active member (a healthy roster / pending entry).
    DuplicatePeer,
    /// The join targets a different run.
    RunIdMismatch,
}

/// A single `tick` output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Output {
    /// A coordinator message to sign + broadcast (`RoundOpen` / `RoundRecord`). Boxed to keep the
    /// enum small.
    Publish(Box<SwarmMessage>),
    /// A non-wire state-change signal.
    Note(Notice),
    /// The input was rejected.
    Reject(Rejection),
}

impl Output {
    /// Convenience constructor for a publish output.
    #[must_use]
    pub fn publish(msg: SwarmMessage) -> Self {
        Output::Publish(Box::new(msg))
    }
}
