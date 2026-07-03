// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Recall path provenance diagnostics — port of `recall_diagnostics.py` (C4).
//!
//! Recall has three silent fallback layers per tier (FTS, vector, and the substring/recency
//! scan). These counters record, per `recall()` call, how many kept rows each path contributed
//! and whether the weak-signal fallback fired, so operators can tell real FTS/vec signal from
//! fallback noise (`recall_diagnostics.py` L1-L47). Python keeps a process-global singleton; here
//! each [`crate::engine::Engine`] owns one instance (one engine per bank ≈ one process in the
//! Python deployment). Diagnostics are read-only signal — they never alter recall behavior.

use std::sync::Mutex;

/// The canonical recall tiers (`recall_diagnostics.py` `RECALL_TIERS` L47).
pub const RECALL_TIERS: [&str; 6] = [
    "wm_fts",
    "wm_vec",
    "wm_fallback",
    "em_fts",
    "em_vec",
    "em_fallback",
];

/// Per-tier counters (`recall_diagnostics.py` `_TierStats` L50-L76).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct TierStats {
    /// Recall invocations that got at least one kept-and-attributed result from this path.
    pub calls_with_hits: u64,
    /// Total kept result rows attributed to this path across all calls (post-filter).
    pub total_hits: u64,
}

#[derive(Debug, Default)]
struct Counters {
    tiers: [TierStats; 6],
    total_calls: u64,
    calls_using_wm_fallback: u64,
    calls_using_em_fallback: u64,
    calls_truly_empty: u64,
}

/// A JSON-serializable snapshot (`recall_diagnostics.py` `snapshot` L172-L228).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Snapshot {
    /// When the counter window opened.
    pub created_at: String,
    /// When this snapshot was taken.
    pub snapshot_at: String,
    /// Outer-call totals and fallback rates.
    pub totals: SnapshotTotals,
    /// Per-tier `(calls_with_hits, total_hits)`, keyed by [`RECALL_TIERS`] order.
    pub by_tier: Vec<(String, TierStats)>,
}

/// The `totals` block of a [`Snapshot`].
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct SnapshotTotals {
    /// Total recall invocations recorded.
    pub calls: u64,
    /// Calls where the working-memory fallback fired.
    pub calls_using_wm_fallback: u64,
    /// Calls where the episodic fallback fired.
    pub calls_using_em_fallback: u64,
    /// Calls that returned zero results from every path including fallback.
    pub calls_truly_empty: u64,
    /// `calls_using_wm_fallback / calls`, clamped to `[0, 1]`.
    pub wm_fallback_rate: f64,
    /// `calls_using_em_fallback / calls`, clamped to `[0, 1]`.
    pub em_fallback_rate: f64,
}

/// Thread-safe recall path counters (`recall_diagnostics.py` `RecallDiagnostics` L79-L242).
#[derive(Debug)]
pub struct RecallDiagnostics {
    counters: Mutex<Counters>,
    created_at: Mutex<String>,
}

impl Default for RecallDiagnostics {
    fn default() -> Self {
        Self {
            counters: Mutex::new(Counters::default()),
            created_at: Mutex::new(crate::util::now_iso()),
        }
    }
}

impl RecallDiagnostics {
    fn tier_index(tier: &str) -> Option<usize> {
        RECALL_TIERS.iter().position(|t| *t == tier)
    }

    /// Record that `tier` contributed `hit_count` kept rows on a recall call
    /// (`record_tier_hits` L116-L128). Unknown tiers are ignored (Python raises; the Rust callers
    /// are the fixed recall pipeline, so a debug log suffices).
    pub fn record_tier_hits(&self, tier: &str, hit_count: usize) {
        let Some(idx) = Self::tier_index(tier) else {
            tracing::debug!(tier, "unknown recall diagnostics tier");
            return;
        };
        let mut c = self.counters.lock().unwrap();
        if hit_count > 0 {
            c.tiers[idx].calls_with_hits += 1;
        }
        c.tiers[idx].total_hits += hit_count as u64;
    }

    /// Record that the WM and/or EM fallback layer fired during a recall call
    /// (`record_fallback_used` L130-L138).
    pub fn record_fallback_used(&self, wm: bool, em: bool) {
        let mut c = self.counters.lock().unwrap();
        if wm {
            c.calls_using_wm_fallback += 1;
        }
        if em {
            c.calls_using_em_fallback += 1;
        }
    }

    /// Record an outer `recall()` invocation; `truly_empty` means zero results from every path
    /// including the fallback (`record_call` L140-L148).
    pub fn record_call(&self, truly_empty: bool) {
        let mut c = self.counters.lock().unwrap();
        c.total_calls += 1;
        if truly_empty {
            c.calls_truly_empty += 1;
        }
    }

    /// Per-tier fraction of calls where the fallback fired, `(wm, em)`, clamped to `[0, 1]`
    /// (`fallback_rate` L150-L170).
    pub fn fallback_rate(&self) -> (f64, f64) {
        let c = self.counters.lock().unwrap();
        if c.total_calls == 0 {
            return (0.0, 0.0);
        }
        let total = c.total_calls as f64;
        (
            (c.calls_using_wm_fallback as f64 / total).min(1.0),
            (c.calls_using_em_fallback as f64 / total).min(1.0),
        )
    }

    /// A JSON-serializable view of the current counters (`snapshot` L172-L228).
    pub fn snapshot(&self) -> Snapshot {
        let c = self.counters.lock().unwrap();
        let total = c.total_calls;
        let (wm_rate, em_rate) = if total == 0 {
            (0.0, 0.0)
        } else {
            (
                (c.calls_using_wm_fallback as f64 / total as f64).min(1.0),
                (c.calls_using_em_fallback as f64 / total as f64).min(1.0),
            )
        };
        Snapshot {
            created_at: self.created_at.lock().unwrap().clone(),
            snapshot_at: crate::util::now_iso(),
            totals: SnapshotTotals {
                calls: total,
                calls_using_wm_fallback: c.calls_using_wm_fallback,
                calls_using_em_fallback: c.calls_using_em_fallback,
                calls_truly_empty: c.calls_truly_empty,
                wm_fallback_rate: wm_rate,
                em_fallback_rate: em_rate,
            },
            by_tier: RECALL_TIERS
                .iter()
                .zip(c.tiers.iter())
                .map(|(name, stats)| ((*name).to_string(), *stats))
                .collect(),
        }
    }

    /// Reset all counters and reopen the measurement window (`reset` L230-L242).
    pub fn reset(&self) {
        *self.counters.lock().unwrap() = Counters::default();
        *self.created_at.lock().unwrap() = crate::util::now_iso();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate_and_rates_clamp() {
        let d = RecallDiagnostics::default();
        d.record_tier_hits("wm_fts", 3);
        d.record_tier_hits("wm_fts", 0);
        d.record_tier_hits("em_vec", 2);
        d.record_fallback_used(true, false);
        d.record_call(false);
        d.record_call(true);

        let snap = d.snapshot();
        assert_eq!(snap.totals.calls, 2);
        assert_eq!(snap.totals.calls_truly_empty, 1);
        assert_eq!(snap.totals.calls_using_wm_fallback, 1);
        assert!((snap.totals.wm_fallback_rate - 0.5).abs() < 1e-9);
        let wm_fts = &snap.by_tier[0];
        assert_eq!(wm_fts.0, "wm_fts");
        assert_eq!(wm_fts.1.calls_with_hits, 1); // the zero-hit call doesn't count
        assert_eq!(wm_fts.1.total_hits, 3);

        d.reset();
        assert_eq!(d.snapshot().totals.calls, 0);
        assert_eq!(d.fallback_rate(), (0.0, 0.0));
    }
}
