// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The ONE place a genesis's coordinator role config is authored (§6.1/§6.2).
//!
//! Every genesis-authoring seat in this crate — the frozen fleet-ceremony authoring
//! ([`crate::ceremony::ceremony_genesis`]), the multi-process acceptance authoring
//! ([`crate::live_genesis::live_genesis`]) and the barrier whole-run harness
//! ([`crate::genesis_run::genesis_envelope`]) — builds the coordinator role's opaque `da_init`
//! config through [`coordinator_role_config`], so the three differ ONLY in the values they pin.
//!
//! # Why the seat exists
//!
//! The coordinator config is where a run's ROUND ARITHMETIC and its CLOCK are decided, and both
//! are relationships, not free values:
//!
//! - `global_batch` is **sequences per round across the assignment roster**
//!   ([`daemon_vhc_proto::envelope::GlobalBatch`]). The coordinator opens
//!   `[data_index, data_index + global_batch)`; each peer's share of it is sliced into
//!   `steps_per_round` inner steps of `micro_batch` sequences by the trainer's own planner, which
//!   refuses an interval the step count does not divide — a window that does not divide plans
//!   ZERO fetches, so the peer trains nothing and never commits.
//! - The phase deadlines (`warmup_s`, `round_train_max_s`, `round_witness_s`, `cooldown_s`) count
//!   ticks of the coordinator guest's logical clock. With `tick_period_ms == 0` that clock is the
//!   deterministic event-driven one — one tick per delivered event, no real timer armed — so a
//!   second-denominated deadline is really a count of events and a quiet run's deadline never
//!   fires at all. Only `tick_period_ms > 0` makes the authored seconds wall-clock seconds.
//!
//! Spelling those relationships once, here, is what keeps an authored round schedule from
//! contradicting the trainer config the same genesis embeds. [`RoundSchedule`] and
//! [`PhaseDeadlines`] validate them at AUTHORING — the established genesis-rule discipline
//! (profile-chunk divisibility, state-chunk validity, cadence↔retention): refuse in the author's
//! hands, never at the first round on the fleet.

use ciborium::value::Value;

use daemon_vhc_proto::envelope::{GlobalBatch, StopCondition};
use daemon_vhc_proto::{CapabilitySet, Hash, Seed, VHC_PROTO_VERSION};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, RunConfig as CoordinatorRunConfig};

/// The deadline value an EVENT-CLOCK run authors: effectively infinite in event counts, i.e. the
/// phase can only ever exit through its event-driven fast path (all-ready / all-committed /
/// all-evidenced). A run without a real tick period may author nothing shorter — a smaller value
/// would read as seconds and behave as a count of delivered events.
pub const EVENT_CLOCK_DEADLINE_S: u64 = 1_000_000;

/// The real-timer period that makes an authored `*_s` deadline mean SECONDS: the coordinator
/// guest advances its logical clock one tick per fired timer, so a 1 s period is one logical
/// second per wall second. Delivered events advance the same clock, so a deadline can only fire
/// EARLIER than its wall value, never later — the conservative direction for a phase ceiling.
pub const WALL_CLOCK_TICK_PERIOD_MS: u64 = 1_000;

/// The genesis-authored coordinator seed (the assignment/committee shuffle salt). One value for
/// every authoring seat: the roster and the window, not the seed, are what the seats vary.
const GENESIS_COORDINATOR_SEED: [u8; 32] = [0x33; 32];

/// A round's sequence schedule: the window the coordinator opens, and the shape the trainers
/// slice it into.
///
/// The window and the trainer's inner loop are ONE relationship. A peer's share of the round
/// window is `global_batch / peers`, which the trainer's planner slices into `steps_per_round`
/// inner steps of `micro_batch` sequences; an indivisible share yields no steps at all. `peers`
/// is the length of the **trainer config's** roster — the assignment denominator is the
/// genesis-pinned roster the trainer plans against, not the coordinator's live membership.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundSchedule {
    /// Sequences per round across the assignment roster (`GlobalBatch.start`/`.end`).
    pub global_batch: u32,
    /// Inner steps per round (the trainer config's `steps_per_round`).
    pub steps_per_round: u32,
    /// Sequences per inner step (the trainer config's `micro_batch`).
    pub micro_batch: u32,
    /// The assignment roster's length — how many ways the window is split.
    pub peers: u32,
}

impl RoundSchedule {
    /// The schedule DERIVED from a trainer config: `steps_per_round × micro_batch × peers`
    /// sequences per round, i.e. exactly one micro-batch per inner step per peer. This is the
    /// smallest window the inner loop can consume, and it cannot contradict the trainer config
    /// because it is computed from it.
    #[must_use]
    pub fn derived(steps_per_round: u32, micro_batch: u32, peers: u32) -> Self {
        Self {
            global_batch: steps_per_round
                .saturating_mul(micro_batch)
                .saturating_mul(peers),
            steps_per_round,
            micro_batch,
            peers,
        }
    }

    /// A schedule whose window is supplied rather than derived (a harness that wants more than
    /// one micro-batch per step). [`RoundSchedule::validate`] still holds it to the trainer
    /// config it will be authored beside.
    #[must_use]
    pub fn explicit(global_batch: u32, steps_per_round: u32, micro_batch: u32, peers: u32) -> Self {
        Self {
            global_batch,
            steps_per_round,
            micro_batch,
            peers,
        }
    }

    /// One peer's assigned sequences per round (`global_batch / peers`, the equal split
    /// `assign_batches` performs over a class-equal roster). Zero when the schedule is invalid.
    #[must_use]
    pub fn sequences_per_peer(&self) -> u32 {
        self.global_batch.checked_div(self.peers).unwrap_or(0)
    }

    /// Refuse a schedule the trainers cannot consume — the zero-step round, made impossible by
    /// construction at authoring.
    ///
    /// # Errors
    /// A human-readable refusal when the roster is empty, any count is zero, the window does not
    /// split evenly across the roster, or a peer's share is not a whole number of inner steps.
    pub fn validate(&self) -> Result<(), String> {
        if self.peers == 0 {
            return Err(
                "the assignment roster is empty: no peer is assigned any of the round window"
                    .to_string(),
            );
        }
        if self.steps_per_round == 0 || self.micro_batch == 0 {
            return Err(format!(
                "the inner loop is empty: steps_per_round {} × micro_batch {} consumes no \
                 sequences",
                self.steps_per_round, self.micro_batch
            ));
        }
        if self.global_batch == 0 {
            return Err("the round window is empty: global_batch is zero".to_string());
        }
        if !self.global_batch.is_multiple_of(self.peers) {
            return Err(format!(
                "the {}-sequence round window does not split evenly across the {}-peer \
                 assignment roster: the peers' shares differ by a sequence and at most one of \
                 them can be a whole number of inner steps",
                self.global_batch, self.peers
            ));
        }
        let per_peer = self.sequences_per_peer();
        if !per_peer.is_multiple_of(self.steps_per_round) {
            return Err(format!(
                "a peer's {per_peer}-sequence window ({} sequences over {} peers) is not \
                 divisible by steps_per_round {}: the round slices into ZERO inner steps, so the \
                 peer plans no fetches, trains nothing and never commits",
                self.global_batch, self.peers, self.steps_per_round
            ));
        }
        Ok(())
    }
}

/// The coordinator's four phase deadlines, in ticks of its logical clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseDeadlines {
    /// The join/warmup wall before the epoch's first round opens.
    pub warmup_s: u64,
    /// The per-round training-phase ceiling.
    pub round_train_max_s: u64,
    /// The witness/finalization wall.
    pub round_witness_s: u64,
    /// The end-of-run cooldown wall.
    pub cooldown_s: u64,
}

impl PhaseDeadlines {
    /// The event-clock shape: every phase exits through its fast path, no deadline is reachable.
    #[must_use]
    pub fn event_clock() -> Self {
        Self {
            warmup_s: EVENT_CLOCK_DEADLINE_S,
            round_train_max_s: EVENT_CLOCK_DEADLINE_S,
            round_witness_s: EVENT_CLOCK_DEADLINE_S,
            cooldown_s: EVENT_CLOCK_DEADLINE_S,
        }
    }

    /// Refuse wall-clock deadlines authored onto a clock that does not measure wall time.
    ///
    /// # Errors
    /// A human-readable refusal when `tick_period_ms == 0` (the deterministic event-driven clock)
    /// and any deadline is short enough to read as seconds.
    pub fn validate(&self, tick_period_ms: u64) -> Result<(), String> {
        if tick_period_ms > 0 {
            return Ok(());
        }
        for (name, value) in [
            ("warmup_s", self.warmup_s),
            ("round_train_max_s", self.round_train_max_s),
            ("round_witness_s", self.round_witness_s),
            ("cooldown_s", self.cooldown_s),
        ] {
            if value < EVENT_CLOCK_DEADLINE_S {
                return Err(format!(
                    "{name} = {value} is a wall-clock deadline, but the coordinator config arms \
                     no real timer (tick_period_ms = 0): its clock advances one tick per \
                     delivered event, so the value counts EVENTS and a quiet phase never times \
                     out at all. Author a real tick period ({WALL_CLOCK_TICK_PERIOD_MS} ms = one \
                     logical second per wall second), or leave every deadline at \
                     {EVENT_CLOCK_DEADLINE_S} (the event-clock shape)"
                ));
            }
        }
        Ok(())
    }
}

/// What an authoring seat pins in the coordinator role's config. Everything not named here is the
/// ratified authoring shape and identical across the seats: no witness committee, no deliberate
/// batch overlap, no verification sampling, no pause/resume principals, and the envelope-hash
/// anchor left zero (a v2 join anchors on the genesis hash, which is the hash of the envelope
/// being authored and so cannot be embedded in it).
pub struct CoordinatorAuthoring<'a> {
    /// The run label (the coordinator's admission id).
    pub run_label: &'a str,
    /// Members required before warmup can end.
    pub min_peers: u32,
    /// Roster ceiling.
    pub max_peers: u32,
    /// Epoch length in rounds (`0` = no epoch boundaries).
    pub epoch_rounds: u64,
    /// Fetch-recovery budget before a stalled peer leaves for the epoch.
    pub stall_rounds_max: u32,
    /// Record absences before a silent member is dropped.
    pub k_absences: u32,
    /// Tokens per sequence (the corpus the run is pinned to).
    pub seq_len: u64,
    /// The round's sequence schedule (validated).
    pub schedule: RoundSchedule,
    /// The phase deadlines (validated against `tick_period_ms`).
    pub deadlines: PhaseDeadlines,
    /// The coordinator's real-timer period in ms; `0` = the deterministic event-driven clock.
    pub tick_period_ms: u64,
    /// The run's stop condition.
    pub stop: StopCondition,
    /// Coordinator-as-storage-client availability verification (§6.4 I6).
    pub verify_availability: bool,
}

/// Author the coordinator role's opaque `da_init` config: the resolved [`CoordinatorState`] plus
/// the two guest-runtime knobs the module reads beside it.
///
/// # Errors
/// A human-readable authoring refusal from [`RoundSchedule::validate`] or
/// [`PhaseDeadlines::validate`], or a CBOR encoding failure.
pub fn coordinator_role_config(authoring: &CoordinatorAuthoring<'_>) -> Result<Value, String> {
    authoring.schedule.validate()?;
    authoring.deadlines.validate(authoring.tick_period_ms)?;

    let run_config = CoordinatorRunConfig {
        run_id: authoring.run_label.to_string(),
        proto_version: VHC_PROTO_VERSION,
        envelope_hash: Hash([0u8; 32]),
        required_capabilities: CapabilitySet::new(),
        min_peers: authoring.min_peers,
        max_peers: authoring.max_peers,
        warmup_s: authoring.deadlines.warmup_s,
        round_train_max_s: authoring.deadlines.round_train_max_s,
        round_witness_s: authoring.deadlines.round_witness_s,
        cooldown_s: authoring.deadlines.cooldown_s,
        epoch_rounds: authoring.epoch_rounds,
        stall_rounds_max: authoring.stall_rounds_max,
        global_batch: GlobalBatch {
            start: authoring.schedule.global_batch,
            end: authoring.schedule.global_batch,
            ramp_rounds: 1,
        },
        stop: authoring.stop,
        steps_per_round: authoring.schedule.steps_per_round,
        seq_len: authoring.seq_len,
        witness_target: 0,
        overlap_bps: 0,
        k_absences: authoring.k_absences,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    let state = CoordinatorState::new(run_config, Seed(GENESIS_COORDINATOR_SEED), 0);
    Ok(Value::Map(vec![
        (
            Value::Text("state".into()),
            Value::serialized(&state).map_err(|e| format!("coordinator state to cbor: {e}"))?,
        ),
        (
            Value::Text("tick_period_ms".into()),
            Value::Integer(authoring.tick_period_ms.into()),
        ),
        (
            Value::Text("verify_availability".into()),
            Value::Bool(authoring.verify_availability),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived schedule is one micro-batch per inner step per peer, and it is sliceable by
    /// construction at any roster size.
    #[test]
    fn a_derived_schedule_is_a_whole_number_of_inner_steps_per_peer() {
        for peers in 1..=8u32 {
            for steps in [1u32, 2, 30] {
                let schedule = RoundSchedule::derived(steps, 1, peers);
                assert_eq!(schedule.global_batch, steps * peers);
                assert_eq!(schedule.sequences_per_peer(), steps);
                schedule
                    .validate()
                    .expect("a derived schedule is sliceable");
            }
        }
    }

    /// The zero-step round is an authoring refusal: a per-peer window the inner-step count does
    /// not divide plans no fetches, so the peer never trains and never commits.
    #[test]
    fn a_window_the_inner_loop_cannot_slice_is_refused() {
        // One sequence per peer against a 30-step inner loop (a round window sized to the roster
        // instead of to the trainer's schedule).
        let err = RoundSchedule::explicit(3, 30, 1, 3)
            .validate()
            .expect_err("an indivisible per-peer window must refuse");
        assert!(err.contains("ZERO inner steps"), "{err}");
        // An unequal split is refused before it can be rounded into one.
        assert!(RoundSchedule::explicit(4, 2, 1, 3).validate().is_err());
        // Empty roster, empty window, empty inner loop.
        assert!(RoundSchedule::derived(30, 1, 0).validate().is_err());
        assert!(RoundSchedule::explicit(0, 30, 1, 3).validate().is_err());
        assert!(RoundSchedule::derived(0, 1, 3).validate().is_err());
    }

    /// Wall-clock deadlines require a wall clock: authoring seconds onto the event-driven clock
    /// is refused, because those seconds would count delivered events.
    #[test]
    fn wall_clock_deadlines_require_a_real_tick_period() {
        let real = PhaseDeadlines {
            warmup_s: 300,
            round_train_max_s: 600,
            round_witness_s: 300,
            cooldown_s: 60,
        };
        let err = real
            .validate(0)
            .expect_err("real seconds on the event clock must refuse");
        assert!(err.contains("tick_period_ms = 0"), "{err}");
        real.validate(WALL_CLOCK_TICK_PERIOD_MS)
            .expect("real seconds under a real tick period");
        PhaseDeadlines::event_clock()
            .validate(0)
            .expect("the event-clock shape needs no timer");
    }

    /// The authored config carries the three keys the coordinator guest reads, and the schedule
    /// it validated.
    #[test]
    fn the_authored_config_carries_the_guests_three_keys() {
        let cfg = coordinator_role_config(&CoordinatorAuthoring {
            run_label: "schedule-shape",
            min_peers: 3,
            max_peers: 3,
            epoch_rounds: 0,
            stall_rounds_max: 4,
            k_absences: 3,
            seq_len: 2048,
            schedule: RoundSchedule::derived(30, 1, 3),
            deadlines: PhaseDeadlines {
                warmup_s: 300,
                round_train_max_s: 600,
                round_witness_s: 300,
                cooldown_s: 60,
            },
            tick_period_ms: WALL_CLOCK_TICK_PERIOD_MS,
            stop: StopCondition::Rounds(48),
            verify_availability: false,
        })
        .expect("author the coordinator config");
        let Value::Map(entries) = &cfg else {
            panic!("the coordinator config is a map");
        };
        let keys: Vec<&str> = entries
            .iter()
            .filter_map(|(k, _)| match k {
                Value::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(keys, ["state", "tick_period_ms", "verify_availability"]);
        let state: CoordinatorState = entries[0].1.clone().deserialized().expect("state decodes");
        assert_eq!(state.config.global_batch.start, 90);
        assert_eq!(state.config.global_batch.end, 90);
        assert_eq!(state.config.steps_per_round, 30);

        // The same authoring with the round window spelled as the peer count instead of derived
        // from the trainer config is refused outright.
        assert!(coordinator_role_config(&CoordinatorAuthoring {
            run_label: "schedule-shape",
            min_peers: 3,
            max_peers: 3,
            epoch_rounds: 0,
            stall_rounds_max: 4,
            k_absences: 3,
            seq_len: 2048,
            schedule: RoundSchedule::explicit(3, 30, 1, 3),
            deadlines: PhaseDeadlines::event_clock(),
            tick_period_ms: WALL_CLOCK_TICK_PERIOD_MS,
            stop: StopCondition::Rounds(48),
            verify_availability: false,
        })
        .is_err());
    }
}
