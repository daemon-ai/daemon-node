// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-ingest` — the reusable **inbound** gate, the symmetric counterpart to `daemon-delivery`
//! (daemon-event-io-spec §5.9.1).
//!
//! `daemon-delivery` stitches the outbound primitives (`delivery_sessions` + `subscribe` +
//! handover-stop) into one turnkey loop so an adapter does not re-implement reply delivery. This
//! crate does the mirror image for the *inbound* side: it owns the transport-agnostic decision of
//! **which** [`AgentCommand`] to submit for an incoming message, given whether the message is
//! *addressed* (a mention / DM / `!command`) and whether the session is currently *busy* — plus the
//! bounded queue/fold buffering — driving [`SessionApi::submit_routed`] so routing, profile binding,
//! and `Primary` seeding stay host-owned.
//!
//! What stays adapter-owned (the input to this crate) is the *transport-specific* classification:
//! parsing a Matrix mention vs. a Slack `@bot` vs. a `!command` prefix. The adapter normalises each
//! transport event into a [`Reception`] (`origin` + `input` + `addressed`); the [`Ingestor`] owns
//! everything transport-agnostic from there.
//!
//! ## Busy tracking
//! The [`Ingestor`] does not open its own subscription. Busy state is driven by two hooks —
//! [`Ingestor::note_turn_started`] / [`Ingestor::note_turn_finished`] — the adapter calls from the
//! outbound stream it *already* consumes via `daemon-delivery` (`TurnStarted` / `TurnFinished` are in
//! the merged log). This keeps the crate host-free, single-subscription, and deterministic to test. A
//! self-subscribing convenience is intentionally out of scope.
//!
//! Reuses only existing wire surface (`submit_routed` + `AgentCommand`), so it carries **no**
//! `WireVersion` / CDDL / MSRV change.

#![forbid(unsafe_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use daemon_api::{ApiError, NodeApi};
use daemon_common::{ReqId, SessionId};
use daemon_protocol::{session_id_for, AgentCommand, IsolationPolicy, Origin, UserMsg};

/// What to do with an **addressed** message that arrives while the session's engine is mid-turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BusyPolicy {
    /// Hold it in a bounded ring and replay it as a `StartTurn` when the current turn finishes
    /// (the default — never drop an addressed message, never interleave two turns).
    #[default]
    Queue,
    /// Interrupt the running turn, then immediately start a new turn from this message.
    Interrupt,
    /// Inject it as mid-turn steering text (`Steer`), drained at the next phase boundary.
    Steer,
}

/// What to do with a **non-addressed** (ambient) message — chatter the agent should *see* as context
/// but not *turn* on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AmbientPolicy {
    /// Append it to the conversation immediately via `AgentCommand::Observe` (folds in when idle,
    /// lands in the following turn when busy — wire v8 semantics). The default now that `Observe`
    /// exists.
    #[default]
    Observe,
    /// Buffer it in a bounded ring and prepend the buffered ambient text, attributed, into the input
    /// of the next addressed `StartTurn` (the no-`Observe` "buffer-and-fold" fallback).
    Fold,
}

/// How an [`Ingestor`] gates inbound messages. `Default` is the recommended shape: queue addressed
/// messages while busy, surface ambient ones via `Observe`, a ~32-entry cap, and `PerThread`
/// isolation (matching `submit_routed`'s own default id derivation).
#[derive(Clone, Copy, Debug)]
pub struct IngestPolicy {
    /// Disposition of an addressed message that arrives mid-turn.
    pub busy: BusyPolicy,
    /// Disposition of a non-addressed (ambient) message.
    pub ambient: AmbientPolicy,
    /// Upper bound on each per-session buffer (the addressed queue and the ambient fold ring); the
    /// oldest entry is evicted when a push would exceed it.
    pub queue_cap: usize,
    /// The isolation this transport derives session ids under — used both to key busy state and so
    /// the gate's id matches the one `submit_routed` resolves for the common (no per-row override)
    /// case.
    pub isolation: IsolationPolicy,
}

impl Default for IngestPolicy {
    fn default() -> Self {
        Self {
            busy: BusyPolicy::default(),
            ambient: AmbientPolicy::default(),
            queue_cap: 32,
            isolation: IsolationPolicy::PerThread,
        }
    }
}

/// A normalised inbound message handed to the [`Ingestor`]. The adapter has already done the
/// transport-specific work of deciding whether the message is `addressed`.
#[derive(Clone, Debug)]
pub struct Reception {
    /// Where it came from (the routing key + reply address).
    pub origin: Origin,
    /// The message body (attribution — who spoke — rides inside the text, adapter-formatted).
    pub input: UserMsg,
    /// Whether this message addresses the agent (mention / DM / `!command`): `true` may open or
    /// steer a turn; `false` is ambient context only.
    pub addressed: bool,
}

/// Per-session gate state: whether a turn is in flight, the ambient fold buffer, and the
/// addressed-while-busy queue. Both buffers are bounded by [`IngestPolicy::queue_cap`].
#[derive(Default)]
struct Gate {
    busy: bool,
    fold: VecDeque<String>,
    queued: VecDeque<String>,
}

fn push_bounded(buf: &mut VecDeque<String>, text: String, cap: usize) {
    if cap == 0 {
        return;
    }
    while buf.len() >= cap {
        buf.pop_front();
    }
    buf.push_back(text);
}

/// Join a drained ambient fold buffer (if any) ahead of `head` into one input body, oldest first.
fn fold_into(fold: &mut VecDeque<String>, head: String) -> String {
    if fold.is_empty() {
        return head;
    }
    let mut parts: Vec<String> = fold.drain(..).collect();
    parts.push(head);
    parts.join("\n")
}

/// The reusable inbound gate. Construct with [`Ingestor::new`] (default policy) or
/// [`Ingestor::with_policy`], feed it [`Reception`]s via [`Ingestor::receive`], and drive its busy
/// state from the outbound turn lifecycle via [`Ingestor::note_turn_started`] /
/// [`Ingestor::note_turn_finished`].
pub struct Ingestor {
    api: Arc<dyn NodeApi>,
    policy: IngestPolicy,
    gates: Mutex<HashMap<SessionId, Gate>>,
    next_req: AtomicU64,
}

impl Ingestor {
    /// An ingestor over `api` with the default [`IngestPolicy`].
    pub fn new(api: Arc<dyn NodeApi>) -> Self {
        Self::with_policy(api, IngestPolicy::default())
    }

    /// An ingestor over `api` with an explicit `policy`.
    pub fn with_policy(api: Arc<dyn NodeApi>, policy: IngestPolicy) -> Self {
        Self {
            api,
            policy,
            gates: Mutex::new(HashMap::new()),
            next_req: AtomicU64::new(1),
        }
    }

    fn req(&self) -> ReqId {
        ReqId(self.next_req.fetch_add(1, Ordering::Relaxed))
    }

    /// Gate one inbound message: decide the command(s) per the policy + current busy state, then
    /// submit them through `submit_routed` (so the host owns routing/profile/Primary). Returns the
    /// [`SessionId`] the origin routed to (for the adapter to map replies back).
    ///
    /// Buffer-only outcomes (ambient in `Fold` mode; addressed-while-busy in `Queue` mode) submit
    /// nothing and return the derived id.
    pub async fn receive(&self, r: Reception) -> Result<SessionId, ApiError> {
        let session = session_id_for(&r.origin, self.policy.isolation);
        // Phase 1: decide + mutate gate state under the lock, collecting commands to submit (the
        // std Mutex must not be held across an await).
        let commands = {
            let mut gates = self.gates.lock().unwrap();
            let gate = gates.entry(session.clone()).or_default();
            self.decide(gate, &r)
        };
        // Phase 2: submit outside the lock.
        for command in commands {
            self.api.submit_routed(r.origin.clone(), command).await?;
        }
        Ok(session)
    }

    /// Decide the commands for `r` against `gate`, mutating buffers/busy as needed.
    fn decide(&self, gate: &mut Gate, r: &Reception) -> Vec<AgentCommand> {
        let cap = self.policy.queue_cap;
        if r.addressed {
            if !gate.busy {
                // Idle: open a turn, folding any buffered ambient context ahead of it.
                let input = fold_into(&mut gate.fold, r.input.text.clone());
                gate.busy = true;
                vec![AgentCommand::StartTurn {
                    input: UserMsg::new(input),
                    request_id: self.req(),
                }]
            } else {
                match self.policy.busy {
                    BusyPolicy::Queue => {
                        push_bounded(&mut gate.queued, r.input.text.clone(), cap);
                        Vec::new()
                    }
                    BusyPolicy::Interrupt => {
                        // Interrupt the running turn, then start the new one (still busy after).
                        let input = fold_into(&mut gate.fold, r.input.text.clone());
                        vec![
                            AgentCommand::Interrupt { reason: None },
                            AgentCommand::StartTurn {
                                input: UserMsg::new(input),
                                request_id: self.req(),
                            },
                        ]
                    }
                    BusyPolicy::Steer => vec![AgentCommand::Steer {
                        text: r.input.text.clone(),
                        request_id: self.req(),
                    }],
                }
            }
        } else {
            match self.policy.ambient {
                AmbientPolicy::Observe => vec![AgentCommand::Observe {
                    input: r.input.clone(),
                    request_id: self.req(),
                }],
                AmbientPolicy::Fold => {
                    push_bounded(&mut gate.fold, r.input.text.clone(), cap);
                    Vec::new()
                }
            }
        }
    }

    /// Mark `session` busy (a turn started). Idempotent. Call from the outbound stream on
    /// `AgentEvent::TurnStarted`.
    pub fn note_turn_started(&self, session: &SessionId) {
        let mut gates = self.gates.lock().unwrap();
        gates.entry(session.clone()).or_default().busy = true;
    }

    /// Mark `session` idle (the turn finished) and, under [`BusyPolicy::Queue`], flush any addressed
    /// messages that arrived mid-turn as a single follow-up `StartTurn` (folding buffered ambient
    /// context ahead of them). Call from the outbound stream on `AgentEvent::TurnFinished`.
    ///
    /// Returns after submitting the flush (if any); the flushed turn re-marks the session busy.
    pub async fn note_turn_finished(&self, session: &SessionId) -> Result<(), ApiError> {
        let flush = {
            let mut gates = self.gates.lock().unwrap();
            let gate = gates.entry(session.clone()).or_default();
            gate.busy = false;
            if gate.queued.is_empty() {
                None
            } else {
                let queued: Vec<String> = gate.queued.drain(..).collect();
                let body = fold_into(&mut gate.fold, queued.join("\n"));
                gate.busy = true;
                Some(AgentCommand::StartTurn {
                    input: UserMsg::new(body),
                    request_id: self.req(),
                })
            }
        };
        if let Some(command) = flush {
            // The session is already resident; submit by its known origin-derived id via submit (no
            // re-routing needed, but submit keeps the same path). We do not have the Origin here, so
            // use the per-session `submit`.
            self.api.submit(session.clone(), command).await?;
        }
        Ok(())
    }
}
