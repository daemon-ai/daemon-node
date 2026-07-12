// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-observe` — the event-sourced run log.
//!
//! A per-run ordered event store plus the projections behind `daemon-cli swarm observe` / `trace`
//! replay (swarm-training-spec.md §14, §10.1; the Psyche event-sourcing lesson, Appendix A.5). It
//! is append-only truth: metrics and the contribution ledger are projections over the log, never
//! primary state.
//!
//! Wave-0 scaffold: only the error type is present; the store + projections land with lane **P**.

#![forbid(unsafe_code)]

/// Errors surfaced by the run-log store and its projections.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ObserveError {
    /// Appending to or reading the ordered event store failed.
    #[error("run-log store error: {0}")]
    Store(String),
    /// Rebuilding a projection from the event log failed.
    #[error("projection error: {0}")]
    Projection(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_renders() {
        assert!(ObserveError::Store("append".into())
            .to_string()
            .contains("run-log store"));
    }
}
