// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The virtual logical clock and one-shot timers (ABI §6.3/§6.5, virtualized SDK-side).
//!
//! Logical time is a `u64` millisecond value, monotone non-decreasing (§6.5). Timers are one-shot,
//! their ids drawn from a per-instance counter starting at 1 and never reused within an instance
//! (§6.3), so a replay of the same event order reproduces every timer id — the deterministic
//! substrate the whole-run transcript relies on.

use std::collections::BTreeSet;

/// A monotone logical millisecond clock (ABI §6.5). Time 0 is run join; it only ever advances.
#[derive(Debug, Clone, Default)]
pub struct VirtualClock {
    now_ms: u64,
}

impl VirtualClock {
    /// A fresh clock at logical time 0 (run join, ABI §6.5).
    #[must_use]
    pub fn new() -> Self {
        Self { now_ms: 0 }
    }

    /// The current slice-constant logical time (ABI §6.5).
    #[must_use]
    pub fn now(&self) -> u64 {
        self.now_ms
    }

    /// Advance to `at`, clamped to be monotone non-decreasing (§6.5 high-water discipline). A
    /// request to move backwards is ignored (the clock stays at its high-water mark).
    pub fn advance_to(&mut self, at: u64) {
        self.now_ms = self.now_ms.max(at);
    }
}

/// The `cancel_timer` outcome (ABI §6.3): the timer had not fired and will not be delivered.
pub const CANCEL_CANCELLED: u32 = daemon_vhc_abi::CANCEL_TIMER_CANCELLED;
/// The `cancel_timer` outcome (ABI §6.3): the timer fired / was delivered / was already cancelled /
/// was never issued.
pub const CANCEL_ALREADY_FIRED_OR_UNKNOWN: u32 =
    daemon_vhc_abi::CANCEL_TIMER_ALREADY_FIRED_OR_UNKNOWN;

/// Per-instance one-shot timer bookkeeping (ABI §6.3): monotone id allocation + cancel semantics.
/// The *scheduling* of when a timer fires lives in the [`crate::sim::Simulator`]'s global queue;
/// this owns only the deterministic id counter and the live-set that makes `cancel`/fire correct.
#[derive(Debug, Clone, Default)]
pub struct Timers {
    next_id: u64,
    /// Armed-but-not-yet-fired timer ids (cancellable). A cancelled or fired id leaves this set.
    live: BTreeSet<u64>,
}

impl Timers {
    /// A fresh per-instance timer table (the counter starts at 1, ABI §6.3).
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            live: BTreeSet::new(),
        }
    }

    /// Arm a one-shot timer, returning its fresh (never-reused) id (ABI §6.3).
    pub fn arm(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.live.insert(id);
        id
    }

    /// Mark `id` fired (delivery is happening now); a subsequent `cancel` returns
    /// `AlreadyFiredOrUnknown`. Returns `true` iff the timer was live (so delivery is legitimate and
    /// must be suppressed if it was cancelled first).
    pub fn fire(&mut self, id: u64) -> bool {
        self.live.remove(&id)
    }

    /// Cancel `id` (ABI §6.3): `CANCEL_CANCELLED` if it was still live (its `Timer` event MUST NOT
    /// be delivered), else `CANCEL_ALREADY_FIRED_OR_UNKNOWN`.
    pub fn cancel(&mut self, id: u64) -> u32 {
        if self.live.remove(&id) {
            CANCEL_CANCELLED
        } else {
            CANCEL_ALREADY_FIRED_OR_UNKNOWN
        }
    }

    /// Whether `id` is still armed (used by the simulator to suppress a queued fire that a later
    /// `cancel` retired, ABI §6.3 "the host MUST NOT deliver its `Timer` event after a `0` return").
    #[must_use]
    pub fn is_live(&self, id: u64) -> bool {
        self.live.contains(&id)
    }
}
