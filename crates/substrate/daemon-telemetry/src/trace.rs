// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Task-local trace scope (elfo's `scope` / `post_recv` pattern).
//!
//! A [`TraceId`] is a correlation handle that must survive every message boundary. The pattern:
//! a task runs *inside* a scope established by [`with_trace`]; anything it sends stamps the
//! scope's current id via [`current_trace`]; when a frame arrives carrying a peer's id, the
//! receiver calls [`set_trace`] to *restore* it into the scope before handling — so work done in
//! response is attributed to the originating trace, across threads, tasks, and process cuts.
//!
//! The scope holds a [`Cell`] so [`set_trace`] can rewrite the current id in place without nesting
//! a new scope per received frame (the long-lived reader-loop case). Outside any scope,
//! [`current_trace`] yields [`TraceId::NONE`] and [`set_trace`] is a silent no-op.

use daemon_common::TraceId;
use std::cell::Cell;
use std::future::Future;

tokio::task_local! {
    static TRACE: Cell<TraceId>;
}

/// Run `fut` inside a fresh trace scope seeded with `id`. Within the scope, [`current_trace`]
/// returns the live value and [`set_trace`] can rewrite it (restore-on-receive).
pub async fn with_trace<F>(id: TraceId, fut: F) -> F::Output
where
    F: Future,
{
    TRACE.scope(Cell::new(id), fut).await
}

/// The current task-local trace id, or [`TraceId::NONE`] if not running inside a scope.
pub fn current_trace() -> TraceId {
    TRACE.try_with(|c| c.get()).unwrap_or(TraceId::NONE)
}

/// Restore/overwrite the current task-local trace id (the receive path). A no-op outside a scope.
pub fn set_trace(id: TraceId) {
    let _ = TRACE.try_with(|c| c.set(id));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scope_carries_and_restores() {
        assert_eq!(current_trace(), TraceId::NONE);

        let t = TraceId(0xABCD);
        with_trace(t, async {
            assert_eq!(current_trace(), t);
            // Restore-on-receive: a frame arrives carrying a different id.
            let peer = TraceId(0x1234);
            set_trace(peer);
            assert_eq!(current_trace(), peer);
        })
        .await;

        // Scope ends; back to none.
        assert_eq!(current_trace(), TraceId::NONE);
    }

    #[tokio::test]
    async fn generate_is_nonzero_and_unique() {
        let a = TraceId::generate();
        let b = TraceId::generate();
        assert!(!a.is_none());
        assert!(!b.is_none());
        assert_ne!(a, b);
    }
}
