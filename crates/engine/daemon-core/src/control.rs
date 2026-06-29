// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`TurnControl`] — the shared, interior-mutable control surface a running turn observes.
//!
//! The live actor ([`crate::actor`]) drives a turn as a future and concurrently services §17
//! control commands. Those commands never touch `&mut engine` (which the turn holds); they mutate
//! this small shared bundle instead, and the turn reads it at phase boundaries (after the model
//! call, after each tool, before finalize). All operations take `&self`, so the actor can pass
//! `&TurnControl` to the turn *and* mutate it from the inbox arm of the same `select!` without a
//! borrow conflict.

use daemon_common::ReqId;
use daemon_protocol::UserMsg;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

/// One queued steer request: the correlation id to echo plus the steering text.
#[derive(Clone, Debug)]
pub struct SteerReq {
    /// Correlation id echoed on [`daemon_protocol::AgentEvent::Steered`].
    pub request_id: ReqId,
    /// The steering text.
    pub text: String,
}

/// The control bundle a turn observes at phase boundaries: cooperative cancellation, a steer queue,
/// and pending snapshot-request ids. Cheaply cloneable (all state is behind `Arc`).
#[derive(Clone, Default)]
pub struct TurnControl {
    cancel: Arc<Mutex<CancellationToken>>,
    steer: Arc<Mutex<VecDeque<SteerReq>>>,
    /// Context-only inbound (`AgentCommand::Observe`) that arrived mid-turn: appended to the
    /// conversation at the next phase boundary as plain user context (no steer marker / `Steered`
    /// ack / re-prompt), so it folds into the following turn the agent runs.
    observe: Arc<Mutex<VecDeque<UserMsg>>>,
    snapshot_req: Arc<Mutex<Vec<ReqId>>>,
}

impl TurnControl {
    /// A fresh control with an un-cancelled token and empty queues.
    pub fn new() -> Self {
        Self::default()
    }

    /// A clone of the current cancellation token (handed to the turn's [`crate::turn::TurnCx`]).
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.lock().unwrap().clone()
    }

    /// Request cooperative cancellation of the in-flight turn.
    pub fn cancel(&self) {
        self.cancel.lock().unwrap().cancel();
    }

    /// Whether cancellation has been requested for the current turn.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.lock().unwrap().is_cancelled()
    }

    /// Replace the cancellation token with a fresh (un-cancelled) one for the next turn.
    pub fn reset(&self) {
        *self.cancel.lock().unwrap() = CancellationToken::new();
    }

    /// Enqueue a steer request (drained at the next phase boundary or to open a steer turn).
    pub fn push_steer(&self, req: SteerReq) {
        self.steer.lock().unwrap().push_back(req);
    }

    /// Drain all queued steer requests in arrival order.
    pub fn drain_steer(&self) -> Vec<SteerReq> {
        self.steer.lock().unwrap().drain(..).collect()
    }

    /// Enqueue a context-only observe (drained at the next phase boundary as plain user context).
    pub fn push_observe(&self, input: UserMsg) {
        self.observe.lock().unwrap().push_back(input);
    }

    /// Drain all queued observe inputs in arrival order.
    pub fn drain_observe(&self) -> Vec<UserMsg> {
        self.observe.lock().unwrap().drain(..).collect()
    }

    /// Record a pending snapshot request (served at the next phase boundary or immediately if idle).
    pub fn push_snapshot(&self, request_id: ReqId) {
        self.snapshot_req.lock().unwrap().push(request_id);
    }

    /// Drain all pending snapshot request ids.
    pub fn drain_snapshot(&self) -> Vec<ReqId> {
        std::mem::take(&mut *self.snapshot_req.lock().unwrap())
    }
}
