// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The rung-3 (api/39) op-id **dispatch context** (spec 09 §10.3, ADR-006).
//!
//! The shared [`dispatch`](crate::dispatch) core binds the current request's client-minted `op_id`
//! (extracted verb-agnostically via [`ApiRequest::op_id`](crate::ApiRequest::op_id)) as a
//! task-local for the duration of the handler, so a node-owned mutation can read it back at the
//! point it owns the change record and stamp uniform `origin_op` provenance — WITHOUT threading an
//! `op_id` parameter through every verb's signature (the anti-per-verb-overfitting discipline,
//! §14.13). Adapters never see this context; the opaque token they round-trip rides the
//! `LifecycleSink::chat_message` seam explicitly, because that report crosses the async serve-loop
//! boundary where a task-local does not reach.
//!
//! Fail-open by construction: outside a bound scope (or when the request carried no `op_id`)
//! [`current_op_id`] is `None`, which is exactly the null-provenance path (`origin_op` absent).

use std::future::Future;

tokio::task_local! {
    static CURRENT_OP_ID: Option<String>;
}

/// Run `fut` with `op_id` bound as the current dispatch op-id context. Within the scope (and any
/// `.await`-inherited frame, not a freshly `spawn`ed task) [`current_op_id`] resolves to `op_id`.
pub async fn with_op_id<F, T>(op_id: Option<String>, fut: F) -> T
where
    F: Future<Output = T>,
{
    CURRENT_OP_ID.scope(op_id, fut).await
}

/// The client-minted `op_id` of the operation currently being dispatched, or `None` when no
/// context is bound / the operation carried no op-id (the null-provenance path).
pub fn current_op_id() -> Option<String> {
    CURRENT_OP_ID.try_with(|op| op.clone()).ok().flatten()
}
