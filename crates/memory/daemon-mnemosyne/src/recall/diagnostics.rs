//! Recall diagnostics — port of `recall_diagnostics.py`.
//!
//! Per-tier hit counters and fallback-rate alarms (`recall_diagnostics.py` L47-L270). Scaffold.

/// The recall tiers tracked (`recall_diagnostics.py` `RECALL_TIERS` L47).
pub const RECALL_TIERS: &[&str] = &[
    "wm_fts",
    "wm_vec",
    "wm_fallback",
    "em_fts",
    "em_vec",
    "em_fallback",
];
