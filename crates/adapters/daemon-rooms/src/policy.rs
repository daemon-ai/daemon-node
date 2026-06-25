//! Floor control — the only genuinely novel logic in the Rooms adapter.
//!
//! Every other piece of a Room reuses the existing routing/ingest/delivery substrate unchanged; the
//! one new decision is *whose* turn it is. Given an inbound Room post, the [`FloorControl`] decides
//! which members may open a turn (per the [`RoomPolicy`] variant), and a [`TurnBudget`] caps how many
//! re-injected turns a single post may cascade into so a `FreeForAll`/`RoundRobin` room cannot echo-
//! storm. All four policies (AddressedOnly / FreeForAll / RoundRobin / Moderator) are decided by
//! [`FloorControl::decide`].

use std::collections::HashSet;

use daemon_protocol::{RoomMember, RoomPolicy};

/// A bounded turn budget guarding against echo storms: a post that re-injects another member's
/// `TurnFinished` as the next post could otherwise cascade unbounded. `max_turns == 0` is unbounded.
#[derive(Clone, Debug)]
pub struct TurnBudget {
    /// The maximum number of turns a single originating post may fan into (`0` = unbounded).
    max_turns: u32,
    /// The number of turns spent against this budget so far.
    used: u32,
}

impl TurnBudget {
    /// A fresh budget capped at `max_turns` (`0` = unbounded).
    pub fn new(max_turns: u32) -> Self {
        Self { max_turns, used: 0 }
    }

    /// Whether another turn may be spent. Unbounded budgets always admit.
    pub fn has_remaining(&self) -> bool {
        self.max_turns == 0 || self.used < self.max_turns
    }

    /// Spend one turn against the budget, returning whether it was admitted.
    pub fn consume(&mut self) -> bool {
        if !self.has_remaining() {
            return false;
        }
        self.used = self.used.saturating_add(1);
        true
    }

    /// Reset the spend counter (a new originating post starts a fresh cascade budget).
    pub fn reset(&mut self) {
        self.used = 0;
    }
}

/// The floor-control engine for one Room: the [`RoomPolicy`] variant plus the cascade [`TurnBudget`].
/// [`FloorControl::decide`] is the gate the room loop consults before fanning a post out to members.
pub struct FloorControl {
    policy: RoomPolicy,
    budget: TurnBudget,
    /// Round-robin cursor: the index of the member whose turn is next (advanced on each grant).
    cursor: usize,
}

impl FloorControl {
    /// A floor-control engine for `policy`, capped at `max_turns` re-injected turns per post.
    pub fn new(policy: RoomPolicy, max_turns: u32) -> Self {
        Self {
            policy,
            budget: TurnBudget::new(max_turns),
            cursor: 0,
        }
    }

    /// The policy this engine enforces.
    pub fn policy(&self) -> &RoomPolicy {
        &self.policy
    }

    /// Decide which members may **open a turn** (`StartTurn`) for a post by `sender` with body `text`,
    /// per the [`RoomPolicy`]; the rest merely observe. Consumes one unit of the cascade [`TurnBudget`]
    /// per granted member, so a bounded budget caps how far a single originating post may cascade
    /// (echo-storm prevention). Returns the admitted member handles (the `sender` is always excluded).
    ///
    /// - `AddressedOnly`: members explicitly mentioned in `text`.
    /// - `FreeForAll`: every other member.
    /// - `RoundRobin`: the next member in a fixed rotation (one per post), advancing the cursor.
    /// - `Moderator { profile }`: the moderator's posts grant the floor to the members it mentions;
    ///   a non-moderator post routes to the moderator (so it arbitrates who speaks next).
    pub fn decide(&mut self, members: &[RoomMember], sender: &str, text: &str) -> HashSet<String> {
        let mut admitted = HashSet::new();
        if !self.budget.has_remaining() {
            return admitted;
        }
        // Clone the policy so the budget/cursor mutations below don't conflict with borrowing `self`.
        let policy = self.policy.clone();
        match &policy {
            RoomPolicy::AddressedOnly => {
                for m in members {
                    if m.member != sender && text.contains(&m.member) && self.budget.consume() {
                        admitted.insert(m.member.clone());
                    }
                }
            }
            RoomPolicy::FreeForAll => {
                for m in members {
                    if m.member != sender && self.budget.consume() {
                        admitted.insert(m.member.clone());
                    }
                }
            }
            RoomPolicy::RoundRobin => {
                let others: Vec<&RoomMember> =
                    members.iter().filter(|m| m.member != sender).collect();
                if !others.is_empty() {
                    let pick = others[self.cursor % others.len()];
                    self.cursor = self.cursor.wrapping_add(1);
                    if self.budget.consume() {
                        admitted.insert(pick.member.clone());
                    }
                }
            }
            RoomPolicy::Moderator { profile } => {
                if sender == profile.as_str() {
                    for m in members {
                        if m.member != sender && text.contains(&m.member) && self.budget.consume() {
                            admitted.insert(m.member.clone());
                        }
                    }
                } else if members.iter().any(|m| &m.member == profile) && self.budget.consume() {
                    admitted.insert(profile.clone());
                }
            }
            // `RoomPolicy` is `#[non_exhaustive]`; a future variant falls back to addressed-only.
            _ => {
                for m in members {
                    if m.member != sender && text.contains(&m.member) && self.budget.consume() {
                        admitted.insert(m.member.clone());
                    }
                }
            }
        }
        admitted
    }

    /// Begin a fresh cascade for a new originating (external/operator) post — resets the budget so the
    /// re-injected replies it triggers draw down a full new allowance.
    pub fn begin_post(&mut self) {
        self.budget.reset();
    }
}
