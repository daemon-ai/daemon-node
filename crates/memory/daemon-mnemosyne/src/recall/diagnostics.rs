// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Recall diagnostics — port of `recall_diagnostics.py`.
//!
//! Per-tier hit counters and fallback-rate alarms (`recall_diagnostics.py` L47-L270) are not wired
//! into recall yet; this module currently centralizes the tier vocabulary so future counters use the
//! same labels as the recall implementation.

/// The recall tiers tracked (`recall_diagnostics.py` `RECALL_TIERS` L47).
pub const RECALL_TIERS: &[&str] = &[
    "wm_fts",
    "wm_vec",
    "wm_fallback",
    "em_fts",
    "em_vec",
    "em_fallback",
];

/// True when `tier` is one of the diagnostic labels this port will track.
pub fn is_recall_tier(tier: &str) -> bool {
    RECALL_TIERS.contains(&tier)
}
