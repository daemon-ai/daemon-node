// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The projection-sync **mutation-effects dispatch context** (daemon-projection-sync-spec.md §4.2).
//!
//! The shared [`dispatch`](crate::dispatch) core scopes a task-local collector around each
//! handler; the feed's `note_change` seam records the exact `(domain, rev)` it assigned at the
//! moment it assigns it — the only accurate source (post-handler rev *sampling* races concurrent
//! mutators, cannot distinguish no-ops, and could attribute another operation's revision). The
//! collected effects power the census check (spec §5: a `MustChange` handler that recorded
//! nothing is a defect) and, later, mutation receipts (spec §9).
//!
//! Mirrors [`op_context`](crate::op_context) exactly: fail-open by construction. Outside a bound
//! scope — internal producers (adapters, workers, ingress) call the same `note_change` seam from
//! spawned tasks — [`record_effect`] is a no-op; their census coverage is asserted by conformance
//! tests, not dispatch.

use std::cell::RefCell;
use std::future::Future;

use crate::DomainRev;

tokio::task_local! {
    static MUTATION_EFFECTS: RefCell<Vec<DomainRev>>;
}

/// Run `fut` with a fresh effects collector bound, returning its output plus every effect
/// recorded within the scope (in recording order; one entry per `note_change`, so a handler
/// touching two domains records two).
pub async fn with_effects<F, T>(fut: F) -> (T, Vec<DomainRev>)
where
    F: Future<Output = T>,
{
    MUTATION_EFFECTS
        .scope(RefCell::new(Vec::new()), async move {
            let out = fut.await;
            let effects = MUTATION_EFFECTS.with(|e| e.borrow().clone());
            (out, effects)
        })
        .await
}

/// Record one mutation effect into the current dispatch scope. A no-op outside a scope (an
/// internal producer's emission) — never an error, so the seam is callable unconditionally from
/// the one place that owns the revision assignment.
pub fn record_effect(effect: DomainRev) {
    let _ = MUTATION_EFFECTS.try_with(|e| e.borrow_mut().push(effect));
}
