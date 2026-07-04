// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Fact-extraction diagnostics — port of `mnemosyne/extraction/diagnostics.py` (C13.b).
//!
//! Pre-C13.b, Python's fact extraction had five silent-failure layers; this counter set records
//! each extraction attempt's outcome per tier so operators can see WHAT is being swallowed.
//! Process-global like Python's singleton (extraction calls fan out from many sites and operators
//! want one aggregate view). Signal-only: never affects extraction behavior.
//!
//! Tier semantics carry over even though the Rust node has a single live transport: everything
//! routes through the injected daemon-core `Provider`, which is the `host` tier. The
//! `remote`/`local`/`cloud` tiers exist for shape parity (a future remote/GGUF fallback slots in
//! without changing the snapshot schema); `wrapper` keeps Python's synthetic tier for failures
//! whose origin can't be attributed post-hoc.

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

/// Canonical extraction tiers (`EXTRACTION_TIERS` L60).
pub const EXTRACTION_TIERS: &[&str] = &["host", "remote", "local", "cloud", "wrapper"];

/// Max recent-error samples kept per tier (`_MAX_ERROR_SAMPLES_PER_TIER` L42).
const MAX_ERROR_SAMPLES_PER_TIER: usize = 10;

/// Cap on raw error message length kept in samples (`_ERROR_MESSAGE_CAP` L47).
const ERROR_MESSAGE_CAP: usize = 200;

/// Sanitize a string for log inclusion (`_safe_for_log` L63-L76): strip control characters and
/// cap length, single line.
pub fn safe_for_log(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_control() || c == '\u{1b}' {
                ' '
            } else {
                c
            }
        })
        .take(200)
        .collect()
}

/// Single-line the message (Python gets this for free from `repr(exc)`; Rust error strings can
/// carry raw newlines) and cap it (`_truncate_error` L119-L125).
fn truncate_error(msg: &str) -> String {
    let flat: String = msg
        .chars()
        .map(|c| {
            if c.is_control() || c == '\u{1b}' {
                ' '
            } else {
                c
            }
        })
        .collect();
    if flat.chars().count() > ERROR_MESSAGE_CAP {
        let head: String = flat.chars().take(ERROR_MESSAGE_CAP).collect();
        format!("{head}...[truncated]")
    } else {
        flat
    }
}

/// Per-tier counters (`_TierStats` L79-L88).
#[derive(Default)]
struct TierStats {
    attempts: u64,
    successes: u64,
    /// Tier ran but returned empty / no parseable facts.
    no_output: u64,
    /// Tier raised an exception.
    failures: u64,
    error_samples: VecDeque<Value>,
}

#[derive(Default)]
struct Totals {
    calls: u64,
    successes: u64,
    failures: u64,
    empty: u64,
}

struct Inner {
    tiers: std::collections::HashMap<&'static str, TierStats>,
    totals: Totals,
    created_at: String,
}

/// Process-global extraction-attempt counters (`ExtractionDiagnostics` L91-L274). All mutations
/// go through one lock; unknown tier names are logged and dropped (Python raises `ValueError` —
/// the Rust seam must never panic inside extraction).
pub struct ExtractionDiagnostics {
    inner: Mutex<Inner>,
}

impl Default for ExtractionDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtractionDiagnostics {
    /// A fresh counter set.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                tiers: EXTRACTION_TIERS
                    .iter()
                    .map(|t| (*t, TierStats::default()))
                    .collect(),
                totals: Totals::default(),
                created_at: crate::util::now_iso(),
            }),
        }
    }

    fn with_tier(&self, tier: &str, f: impl FnOnce(&mut TierStats)) {
        let mut inner = self.inner.lock().unwrap();
        match EXTRACTION_TIERS.iter().find(|t| **t == tier) {
            Some(key) => f(inner.tiers.get_mut(key).expect("tier pre-seeded")),
            None => tracing::warn!(tier, "unknown extraction tier; dropping record"),
        }
    }

    /// Record that an extraction attempt at `tier` is starting (`record_attempt`).
    pub fn record_attempt(&self, tier: &str) {
        self.with_tier(tier, |s| s.attempts += 1);
    }

    /// Record that `tier` returned non-empty facts (`record_success`).
    pub fn record_success(&self, tier: &str) {
        self.with_tier(tier, |s| s.successes += 1);
    }

    /// Record that `tier` ran without error but returned no parseable facts (`record_no_output`)
    /// — "LLM said nothing" vs "LLM crashed" triage differently.
    pub fn record_no_output(&self, tier: &str) {
        self.with_tier(tier, |s| s.no_output += 1);
    }

    /// Record that `tier` failed, with an error message and/or a named reason
    /// (`record_failure` — e.g. `json_parse_failed`, `timeout_or_backend_error`).
    pub fn record_failure(&self, tier: &str, error: Option<&str>, reason: Option<&str>) {
        self.with_tier(tier, |s| {
            s.failures += 1;
            let mut sample = json!({"at": crate::util::now_iso()});
            match (error, reason) {
                (Some(e), _) => {
                    sample["type"] = json!("error");
                    sample["msg"] = json!(truncate_error(e));
                }
                (None, Some(r)) => {
                    sample["type"] = json!("reason");
                    sample["msg"] = json!(truncate_error(r));
                }
                (None, None) => {
                    sample["type"] = json!("unspecified");
                    sample["msg"] = json!("");
                }
            }
            if let Some(r) = reason {
                sample["reason"] = json!(r);
            }
            s.error_samples.push_back(sample);
            if s.error_samples.len() > MAX_ERROR_SAMPLES_PER_TIER {
                s.error_samples.pop_front();
            }
        });
    }

    /// Record the outcome of an outer extraction call, once per invocation (`record_call`):
    /// `succeeded` = at least one tier returned facts; `all_empty` = every tier ran clean but
    /// returned nothing; neither = hard failure.
    pub fn record_call(&self, succeeded: bool, all_empty: bool) {
        let mut inner = self.inner.lock().unwrap();
        inner.totals.calls += 1;
        if succeeded {
            inner.totals.successes += 1;
        } else if all_empty {
            inner.totals.empty += 1;
        } else {
            inner.totals.failures += 1;
        }
    }

    /// Fraction of outer calls that returned facts; 0.0 before any call (`success_rate`).
    pub fn success_rate(&self) -> f64 {
        let inner = self.inner.lock().unwrap();
        if inner.totals.calls == 0 {
            0.0
        } else {
            inner.totals.successes as f64 / inner.totals.calls as f64
        }
    }

    /// JSON-serializable view of the current state (`snapshot` L210-L261 shape).
    pub fn snapshot(&self) -> Value {
        let inner = self.inner.lock().unwrap();
        let by_tier: serde_json::Map<String, Value> = EXTRACTION_TIERS
            .iter()
            .map(|tier| {
                let s = &inner.tiers[tier];
                (
                    tier.to_string(),
                    json!({
                        "attempts": s.attempts,
                        "successes": s.successes,
                        "no_output": s.no_output,
                        "failures": s.failures,
                        "error_samples": s.error_samples.iter().cloned().collect::<Vec<_>>(),
                    }),
                )
            })
            .collect();
        let rate = if inner.totals.calls == 0 {
            0.0
        } else {
            inner.totals.successes as f64 / inner.totals.calls as f64
        };
        json!({
            "created_at": inner.created_at,
            "snapshot_at": crate::util::now_iso(),
            "totals": {
                "calls": inner.totals.calls,
                "successes": inner.totals.successes,
                "failures": inner.totals.failures,
                "empty": inner.totals.empty,
                "success_rate": rate,
            },
            "by_tier": by_tier,
        })
    }

    /// Reset all counters and `created_at` (`reset`).
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.tiers = EXTRACTION_TIERS
            .iter()
            .map(|t| (*t, TierStats::default()))
            .collect();
        inner.totals = Totals::default();
        inner.created_at = crate::util::now_iso();
    }
}

/// The process-global instance, lazily initialized (`get_diagnostics` L283-L294).
pub fn get_diagnostics() -> &'static ExtractionDiagnostics {
    static SINGLETON: OnceLock<ExtractionDiagnostics> = OnceLock::new();
    SINGLETON.get_or_init(ExtractionDiagnostics::new)
}

/// Convenience snapshot of the process-global diagnostics (`get_extraction_stats`).
pub fn get_extraction_stats() -> Value {
    get_diagnostics().snapshot()
}

/// Convenience reset of the process-global diagnostics (`reset_extraction_stats`).
pub fn reset_extraction_stats() {
    get_diagnostics().reset()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_and_snapshot_shape() {
        let d = ExtractionDiagnostics::new();
        d.record_attempt("host");
        d.record_success("host");
        d.record_call(true, false);
        d.record_attempt("host");
        d.record_no_output("host");
        d.record_call(false, true);
        d.record_attempt("host");
        d.record_failure(
            "host",
            Some("boom\nline2"),
            Some("timeout_or_backend_error"),
        );
        d.record_call(false, false);

        let snap = d.snapshot();
        assert_eq!(snap["totals"]["calls"], 3);
        assert_eq!(snap["totals"]["successes"], 1);
        assert_eq!(snap["totals"]["empty"], 1);
        assert_eq!(snap["totals"]["failures"], 1);
        assert!((snap["totals"]["success_rate"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-9);
        let host = &snap["by_tier"]["host"];
        assert_eq!(host["attempts"], 3);
        assert_eq!(host["successes"], 1);
        assert_eq!(host["no_output"], 1);
        assert_eq!(host["failures"], 1);
        let sample = &host["error_samples"][0];
        assert_eq!(sample["reason"], "timeout_or_backend_error");
        assert!(
            !sample["msg"].as_str().unwrap().contains('\n'),
            "control chars sanitized: {sample}"
        );
        // All five tiers present even when untouched.
        for tier in EXTRACTION_TIERS {
            assert!(snap["by_tier"].get(*tier).is_some(), "{tier} missing");
        }
    }

    #[test]
    fn unknown_tier_is_dropped_not_panicking() {
        let d = ExtractionDiagnostics::new();
        d.record_attempt("carrier-pigeon");
        assert_eq!(d.snapshot()["by_tier"]["host"]["attempts"], 0);
    }

    #[test]
    fn error_samples_are_bounded() {
        let d = ExtractionDiagnostics::new();
        for i in 0..15 {
            d.record_failure("cloud", None, Some(&format!("r{i}")));
        }
        let snap = d.snapshot();
        let samples = snap["by_tier"]["cloud"]["error_samples"]
            .as_array()
            .unwrap();
        assert_eq!(samples.len(), 10, "bounded at {MAX_ERROR_SAMPLES_PER_TIER}");
        assert_eq!(samples[0]["reason"], "r5", "oldest evicted");
    }

    #[test]
    fn reset_zeroes_everything() {
        let d = ExtractionDiagnostics::new();
        d.record_attempt("host");
        d.record_call(true, false);
        d.reset();
        assert_eq!(d.snapshot()["totals"]["calls"], 0);
        assert_eq!(d.success_rate(), 0.0);
    }

    #[test]
    fn long_error_messages_truncate() {
        let d = ExtractionDiagnostics::new();
        d.record_failure("wrapper", Some(&"x".repeat(500)), None);
        let snap = d.snapshot();
        let msg = snap["by_tier"]["wrapper"]["error_samples"][0]["msg"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(msg.ends_with("...[truncated]"));
        assert!(msg.len() <= 200 + "...[truncated]".len());
    }
}
