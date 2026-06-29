// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The deterministic skill curator (hermes `curator.py`, minus the LLM consolidation fork).
//!
//! Library hygiene as a pure transition table over the per-profile usage sidecar: an agent-created
//! skill that goes unused past a staleness threshold is marked `Stale`, and one unused past a longer
//! threshold is `Archived` (moved out of discovery). Activity (a `skill_view`/patch) reactivates a
//! stale skill. Pinned, operator-authored (`User`), and binary-bundled skills are never touched.
//!
//! [`apply_automatic_transitions`] is pure (no I/O, `now` injected) so the transition table is unit
//! testable; the caller applies each transition's side effects (usage `set_state` + the physical
//! `SkillStore::archive`/`restore`).

use std::collections::BTreeMap;

use daemon_common::{SkillCreator, SkillState, SkillUsage};

/// Days-to-milliseconds.
const DAY_MS: u64 = 24 * 60 * 60 * 1000;

/// The curator's thresholds (hermes defaults: stale after 30d, archive after 90d).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CuratorConfig {
    /// Idle days after which an active agent-created skill becomes `Stale`.
    pub stale_after_days: u64,
    /// Idle days after which a skill becomes `Archived` (moved out of discovery).
    pub archive_after_days: u64,
}

impl Default for CuratorConfig {
    fn default() -> Self {
        Self {
            stale_after_days: 30,
            archive_after_days: 90,
        }
    }
}

/// One proposed state change for a skill (the caller applies the side effects).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CuratorTransition {
    /// The skill (bundle) name.
    pub name: String,
    /// The state it is moving from.
    pub from: SkillState,
    /// The state it should move to.
    pub to: SkillState,
}

/// Whether a usage record is eligible for automatic curation: only agent-created, unpinned skills
/// are auto-archived (operator-authored and binary-bundled skills are protected — the latter is also
/// rejected physically by [`crate::SkillStore::archive`]).
fn eligible(usage: &SkillUsage) -> bool {
    usage.created_by == SkillCreator::Agent && !usage.pinned
}

/// Compute the automatic lifecycle transitions for a profile's skills at time `now_ms`.
///
/// - active + idle past `stale_after_days` -> `Stale`
/// - (active|stale) + idle past `archive_after_days` -> `Archived`
/// - (stale) + recent activity (within the stale window) -> `Active` (reactivation)
///
/// Ineligible skills (pinned / user / bundled) and skills already in their target state produce no
/// transition. `Archived` skills are not auto-reactivated here (they are out of discovery, so they
/// cannot accrue activity; restore is explicit).
pub fn apply_automatic_transitions(
    usage: &BTreeMap<String, SkillUsage>,
    now_ms: u64,
    cfg: CuratorConfig,
) -> Vec<CuratorTransition> {
    let stale_ms = cfg.stale_after_days.saturating_mul(DAY_MS);
    let archive_ms = cfg.archive_after_days.saturating_mul(DAY_MS);
    let mut out = Vec::new();
    for (name, u) in usage {
        if !eligible(u) {
            continue;
        }
        let idle = now_ms.saturating_sub(u.last_activity_ms());
        let target = if idle >= archive_ms {
            SkillState::Archived
        } else if idle >= stale_ms {
            SkillState::Stale
        } else {
            SkillState::Active
        };
        // No-op when already in the target state. Never auto-reactivate an archived skill (its body
        // is out of discovery; restoring it is an explicit operator/curator action).
        if target == u.state || (u.state == SkillState::Archived && target != SkillState::Archived)
        {
            continue;
        }
        out.push(CuratorTransition {
            name: name.clone(),
            from: u.state,
            to: target,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(state: SkillState, last_activity_ms: u64) -> SkillUsage {
        SkillUsage {
            created_by: SkillCreator::Agent,
            state,
            last_used_ms: Some(last_activity_ms),
            ..SkillUsage::default()
        }
    }

    fn at_days(d: u64) -> u64 {
        d * DAY_MS
    }

    #[test]
    fn active_goes_stale_then_archived() {
        let cfg = CuratorConfig::default();
        let now = at_days(200);
        let mut map = BTreeMap::new();
        map.insert("fresh".into(), agent(SkillState::Active, at_days(195))); // 5d idle
        map.insert("aging".into(), agent(SkillState::Active, at_days(160))); // 40d idle
        map.insert("ancient".into(), agent(SkillState::Active, at_days(50))); // 150d idle
        let t = apply_automatic_transitions(&map, now, cfg);
        assert_eq!(t.len(), 2);
        assert!(t
            .iter()
            .any(|x| x.name == "aging" && x.to == SkillState::Stale));
        assert!(t
            .iter()
            .any(|x| x.name == "ancient" && x.to == SkillState::Archived));
    }

    #[test]
    fn stale_reactivates_on_recent_activity() {
        let cfg = CuratorConfig::default();
        let now = at_days(200);
        let mut map = BTreeMap::new();
        map.insert("revived".into(), agent(SkillState::Stale, at_days(199))); // 1d idle
        let t = apply_automatic_transitions(&map, now, cfg);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].to, SkillState::Active);
    }

    #[test]
    fn pinned_user_and_archived_are_protected() {
        let cfg = CuratorConfig::default();
        let now = at_days(500);
        let mut map = BTreeMap::new();
        let mut pinned = agent(SkillState::Active, 0);
        pinned.pinned = true;
        map.insert("pinned".into(), pinned);
        let mut user = agent(SkillState::Active, 0);
        user.created_by = SkillCreator::User;
        map.insert("user".into(), user);
        map.insert("already".into(), agent(SkillState::Archived, 0));
        let t = apply_automatic_transitions(&map, now, cfg);
        assert!(t.is_empty(), "no transitions, got {t:?}");
    }
}
