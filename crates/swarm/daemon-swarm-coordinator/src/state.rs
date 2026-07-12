// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator state (spec §6.2) — the value `tick` transforms.
//!
//! Everything here is `serde`-serializable so [`CoordinatorState`] round-trips through canonical CBOR
//! byte-identically (the replay-oracle foundation, I1 / TDD PROTO-20). Rounds are held in a fixed
//! ring of [`NUM_STORED_ROUNDS`] slots (Psyche's shape, Appendix A.1), threading `data_index` and the
//! seed from round to round (TDD PROTO-3).

use std::collections::BTreeMap;

use daemon_swarm_proto::messages::{
    Attestation, BatchWindow, Commitment, RecordEntry, RoundRecord, ThroughputClass,
};
use daemon_swarm_proto::{IrohId, PeerId, Seed, StateDigest};
use serde::{Deserialize, Serialize};

use crate::config::RunConfig;

/// The fixed ring of stored rounds (Psyche `NUM_STORED_ROUNDS`, Appendix A.1) — absorbs out-of-order
/// arrivals and the stall ladder (§6.2).
pub const NUM_STORED_ROUNDS: usize = 4;

/// The run lifecycle phase (spec §6.2; Psyche `RunState`, Appendix A.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Pre-initialization (halted).
    Uninitialized,
    /// Gathering the `min_peers` floor (§6.2).
    WaitingForMembers,
    /// Roster frozen for the epoch; peers report model-ready (§6.2).
    Warmup,
    /// A round is open for training + exchange (§6.4).
    RoundTrain,
    /// Grace for straggling commitments / attestations (§6.4).
    RoundWitness,
    /// Epoch end / stop reached — checkpoint window (§6.2).
    Cooldown,
    /// Terminal (halted): `[data].stop` reached (§6.2).
    Finished,
    /// Operator-paused (halted); only an authorized principal resumes (§11.1).
    Paused,
}

impl Phase {
    /// Whether `tick` treats this phase as halted (`Uninitialized`/`Finished`/`Paused`; PROTO-14).
    #[must_use]
    pub fn is_halted(self) -> bool {
        matches!(self, Phase::Uninitialized | Phase::Finished | Phase::Paused)
    }

    /// Whether round evidence (commitments/attestations/receipts) is meaningful in this phase.
    #[must_use]
    pub fn is_round_active(self) -> bool {
        matches!(self, Phase::RoundTrain | Phase::RoundWitness)
    }
}

/// A roster member's liveness state (Psyche `ClientState`, Appendix A.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientState {
    /// Participating normally.
    Healthy,
    /// Dropped after K record-absences (§6.4); re-joinable at the next epoch.
    Dropped,
}

/// A roster member (§6.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Member {
    /// The node identity (ed25519 public key, §7.2).
    pub peer: PeerId,
    /// The peer's iroh `NodeId` bound at join (§7.2).
    pub iroh_id: IrohId,
    /// Declared throughput class (seeds assignment weights, §6.3).
    pub class: ThroughputClass,
    /// Liveness state.
    pub state: ClientState,
    /// Epoch at which the member (re)joined.
    pub joined_epoch: u64,
    /// Consecutive record-absences (§6.4; reset on presence in a record).
    pub absences: u32,
    /// Highest round the member has heartbeated (liveness).
    pub last_seen_round: u64,
    /// Round of the member's most recent `Straggle` (stall-window accounting, PROTO-7).
    pub last_straggle_round: Option<u64>,
}

impl Member {
    /// A fresh healthy member joining at `epoch`.
    #[must_use]
    pub fn joining(peer: PeerId, iroh_id: IrohId, class: ThroughputClass, epoch: u64) -> Self {
        Self {
            peer,
            iroh_id,
            class,
            state: ClientState::Healthy,
            joined_epoch: epoch,
            absences: 0,
            last_seen_round: 0,
            last_straggle_round: None,
        }
    }

    /// Whether the member is currently healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.state == ClientState::Healthy
    }
}

/// Accumulated evidence for one round slot (§6.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundState {
    /// Round number (`height`).
    pub round: u64,
    /// Round seed.
    pub seed: Seed,
    /// Data cursor at round open.
    pub data_index: u64,
    /// The round's global batch window.
    pub batch: BatchWindow,
    /// The selected witness set for the round (§6.3).
    pub witnesses: Vec<PeerId>,
    /// Trainer commitments, by peer.
    pub commitments: BTreeMap<PeerId, Commitment>,
    /// Aggregated storage-receipt verified entries (availability evidence, §6.4 I6).
    pub receipts: Vec<RecordEntry>,
    /// Witness attestations, by witness.
    pub attestations: BTreeMap<PeerId, Attestation>,
    /// Post-ingest state digests, by peer (§5.6).
    pub digests: BTreeMap<PeerId, StateDigest>,
    /// The frozen consensus record, once committed.
    pub record: Option<RoundRecord>,
    /// Set when two peers report divergent digests for this round (§6.4).
    pub desync: bool,
}

impl RoundState {
    /// An empty placeholder slot.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            round: 0,
            seed: Seed([0; 32]),
            data_index: 0,
            batch: BatchWindow { start: 0, end: 0 },
            witnesses: Vec::new(),
            commitments: BTreeMap::new(),
            receipts: Vec::new(),
            attestations: BTreeMap::new(),
            digests: BTreeMap::new(),
            record: None,
            desync: false,
        }
    }

    /// A fresh slot opened for `round`.
    #[must_use]
    pub fn opened(
        round: u64,
        seed: Seed,
        data_index: u64,
        batch: BatchWindow,
        witnesses: Vec<PeerId>,
    ) -> Self {
        let mut s = Self::empty();
        s.round = round;
        s.seed = seed;
        s.data_index = data_index;
        s.batch = batch;
        s.witnesses = witnesses;
        s
    }
}

/// The fixed ring of stored rounds (§6.2, PROTO-3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundRing {
    /// The stored slots (always [`NUM_STORED_ROUNDS`] long).
    pub slots: Vec<RoundState>,
}

impl Default for RoundRing {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRing {
    /// A ring of empty slots.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: vec![RoundState::empty(); NUM_STORED_ROUNDS],
        }
    }

    /// The ring index for `round` (`round mod NUM_STORED_ROUNDS`).
    #[must_use]
    pub fn index_of(round: u64) -> usize {
        (round % NUM_STORED_ROUNDS as u64) as usize
    }

    /// The slot currently holding `round`, if the ring still stores it.
    #[must_use]
    pub fn get(&self, round: u64) -> Option<&RoundState> {
        let s = &self.slots[Self::index_of(round)];
        (s.round == round).then_some(s)
    }

    /// A mutable borrow of `round`'s slot iff the slot currently holds `round`.
    pub fn get_mut(&mut self, round: u64) -> Option<&mut RoundState> {
        let slot = &mut self.slots[Self::index_of(round)];
        if slot.round == round {
            Some(slot)
        } else {
            None
        }
    }

    /// Overwrite `round`'s slot (ring reuse, PROTO-3 wrap).
    pub fn install(&mut self, rs: RoundState) {
        let idx = Self::index_of(rs.round);
        self.slots[idx] = rs;
    }
}

/// The full coordinator state — the value `tick(state, input) -> (state', outputs)` transforms.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinatorState {
    /// The resolved run configuration.
    pub config: RunConfig,
    /// Current lifecycle phase.
    pub phase: Phase,
    /// Current epoch (roster-stable span).
    pub epoch: u64,
    /// Current round number (`height`).
    pub round: u64,
    /// Data cursor (sequences consumed so far).
    pub data_index: u64,
    /// Current round seed.
    pub seed: Seed,
    /// The roster (healthy + dropped members).
    pub roster: Vec<Member>,
    /// Joins admitted mid-epoch, applied at the next `WaitingForMembers` (§6.2).
    pub pending: Vec<Member>,
    /// The stored-rounds ring.
    pub rounds: RoundRing,
    /// Unix seconds at which the current phase began.
    pub phase_start_s: u64,
    /// The last observed clock (time enters only as `Input::Clock`).
    pub now_s: u64,
    /// The round at which the current epoch's training began (epoch-boundary accounting).
    pub epoch_start_round: u64,
    /// Cumulative tokens applied (for `stop = { tokens }`).
    pub tokens_done: u64,
    /// Cumulative committed rounds (for `stop = { rounds }`).
    pub rounds_done: u64,
    /// The phase paused from (for context; resume returns to `WaitingForMembers`).
    pub paused_from: Option<Phase>,
    /// Highest round any peer has reported (epoch global-lead disjunct, PROTO-17).
    pub max_reported_round: u64,
}

impl CoordinatorState {
    /// A new run in `WaitingForMembers` at `now_s`, seeded with `initial_seed`.
    #[must_use]
    pub fn new(config: RunConfig, initial_seed: Seed, now_s: u64) -> Self {
        Self {
            config,
            phase: Phase::WaitingForMembers,
            epoch: 0,
            round: 0,
            data_index: 0,
            seed: initial_seed,
            roster: Vec::new(),
            pending: Vec::new(),
            rounds: RoundRing::new(),
            phase_start_s: now_s,
            now_s,
            epoch_start_round: 0,
            tokens_done: 0,
            rounds_done: 0,
            paused_from: None,
            max_reported_round: 0,
        }
    }

    /// The healthy roster members.
    pub fn healthy(&self) -> impl Iterator<Item = &Member> {
        self.roster.iter().filter(|m| m.is_healthy())
    }

    /// The count of healthy members.
    #[must_use]
    pub fn healthy_count(&self) -> u32 {
        self.healthy().count() as u32
    }

    /// The sorted healthy peer ids (canonical order for committee/roster digest).
    #[must_use]
    pub fn healthy_peer_ids(&self) -> Vec<PeerId> {
        let mut v: Vec<PeerId> = self.healthy().map(|m| m.peer).collect();
        v.sort_unstable();
        v
    }

    /// A mutable borrow of the roster member with `peer`, if present.
    pub fn member_mut(&mut self, peer: &PeerId) -> Option<&mut Member> {
        self.roster.iter_mut().find(|m| &m.peer == peer)
    }

    /// Whether `peer` is a healthy roster member.
    #[must_use]
    pub fn is_healthy_member(&self, peer: &PeerId) -> bool {
        self.roster
            .iter()
            .any(|m| &m.peer == peer && m.is_healthy())
    }
}
