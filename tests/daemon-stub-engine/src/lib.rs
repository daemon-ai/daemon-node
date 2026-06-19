//! `daemon-stub-engine` — a minimal §17-speaking engine for exercising the substrate.
//!
//! Implements the protocol-agnostic [`Incarnation`] seam from `daemon-activation`, but does so by
//! speaking the real (minimal) §17 surface internally: it consumes [`AgentCommand`]s, emits
//! [`AgentEvent`]s with a monotonic `seq`, produces a typed [`Snapshot`] at the suspend boundary,
//! and applies background completions idempotently on rehydration. It lets `daemon-conformance`
//! drive the durable activation core without the real `daemon-core`.
//!
//! Behaviour of one session: on first activation it delegates exactly one background job and
//! suspends; on the completion-triggered rehydration it applies the completion and finishes.

#![forbid(unsafe_code)]

use daemon_activation::{EngineError, EngineFactory, Incarnation, SnapshotBlob, Step};
use daemon_common::{Epoch, JobId};
use daemon_protocol::{
    AgentCommand, AgentEvent, CompletionSource, EndReason, ReqId, Snapshot, TurnSummary,
    TurnTrigger, UserMsg,
};
use daemon_store::{JobCommand, JobCompletion};

/// Internal decision produced while the snapshot is borrowed mutably, applied (with §17 events)
/// after that borrow ends.
enum StepDecision {
    Completed,
    Suspended { job_id: JobId, epoch: Epoch },
}

/// Builds fresh [`StubIncarnation`]s for the activation layer.
#[derive(Default)]
pub struct StubEngineFactory;

impl StubEngineFactory {
    /// Construct the factory.
    pub fn new() -> Self {
        Self
    }
}

impl EngineFactory for StubEngineFactory {
    fn create(&self) -> Box<dyn Incarnation> {
        Box::new(StubIncarnation::default())
    }
}

/// A single deterministic engine incarnation.
#[derive(Default)]
pub struct StubIncarnation {
    snapshot: Option<Snapshot>,
    unapplied: Vec<JobCompletion>,
    seq: u64,
    /// The §17 events emitted this activation (kept so the stub demonstrably speaks the protocol).
    events: Vec<AgentEvent>,
}

impl StubIncarnation {
    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    fn emit(&mut self, make: impl FnOnce(u64) -> AgentEvent) {
        let seq = self.next_seq();
        self.events.push(make(seq));
    }

    /// The §17 events emitted during the last [`Incarnation::run`] (test observability).
    pub fn events(&self) -> &[AgentEvent] {
        &self.events
    }
}

#[async_trait::async_trait]
impl Incarnation for StubIncarnation {
    async fn hydrate(
        &mut self,
        snapshot: SnapshotBlob,
        unapplied: Vec<JobCompletion>,
    ) -> Result<(), EngineError> {
        let snapshot = Snapshot::decode(&snapshot)?;
        self.snapshot = Some(snapshot);
        self.unapplied = unapplied;
        Ok(())
    }

    async fn run(&mut self) -> Result<Step, EngineError> {
        let session = self
            .snapshot
            .as_ref()
            .ok_or_else(|| EngineError::Other("run before hydrate".into()))?
            .session_id
            .clone();

        // The host would deliver this StartTurn over §17; we model consuming it.
        let _command = AgentCommand::StartTurn {
            input: UserMsg::new("(stub turn)"),
            request_id: ReqId(self.seq),
        };

        let resuming = !self.unapplied.is_empty();
        let trigger = if resuming {
            TurnTrigger::BackgroundCompletion {
                source: CompletionSource::Delegation(self.unapplied[0].job_id.clone()),
            }
        } else {
            TurnTrigger::User
        };
        self.emit(|seq| AgentEvent::TurnStarted { seq, trigger });

        // Decide the step and mutate the snapshot in a scope that ends before we emit again
        // (keeps the `&mut snapshot` borrow disjoint from `self.emit`).
        let unapplied_count = self.unapplied.len();
        let step = {
            let snap = self.snapshot.as_mut().unwrap();

            // Case A: a background completion arrived — apply it idempotently and finish.
            if unapplied_count > 0 {
                for _ in 0..unapplied_count {
                    snap.conversation.push("tool", "background work complete");
                }
                snap.waiting_for.clear();
                snap.epoch = snap.epoch.next();
                StepDecision::Completed
            }
            // Case B: first activation — delegate exactly one background job and suspend.
            else if snap.waiting_for.is_empty() && snap.conversation.is_empty() {
                snap.epoch = snap.epoch.next();
                let job_id = JobId::new(format!("{session}:{}:job", snap.epoch.0));
                snap.waiting_for.push(job_id.clone());
                snap.conversation
                    .push("assistant", "delegating background work");
                StepDecision::Suspended {
                    job_id,
                    epoch: snap.epoch,
                }
            }
            // Case C: re-activated while still suspended (e.g. recovery before the worker ran) —
            // re-suspend with the same deterministic job (the outbox dedupes the re-enqueue).
            else if let Some(job_id) = snap.waiting_for.first().cloned() {
                StepDecision::Suspended {
                    job_id,
                    epoch: snap.epoch,
                }
            }
            // Case D: nothing outstanding — terminal.
            else {
                StepDecision::Completed
            }
        };

        match step {
            StepDecision::Completed => {
                self.emit(|seq| AgentEvent::TurnFinished {
                    seq,
                    summary: TurnSummary {
                        end_reason: EndReason::Completed,
                    },
                });
                Ok(Step::Completed)
            }
            StepDecision::Suspended { job_id, epoch } => {
                self.emit(|seq| AgentEvent::TurnFinished {
                    seq,
                    summary: TurnSummary {
                        end_reason: EndReason::Suspended,
                    },
                });
                Ok(Step::Suspended {
                    job: JobCommand {
                        job_id,
                        session_id: session,
                        epoch,
                        payload: b"stub-work".to_vec(),
                    },
                })
            }
        }
    }

    fn checkpoint(&self) -> Result<SnapshotBlob, EngineError> {
        let snap = self
            .snapshot
            .as_ref()
            .ok_or_else(|| EngineError::Other("checkpoint before hydrate".into()))?;
        Ok(snap.encode()?)
    }

    fn epoch(&self) -> Epoch {
        self.snapshot.as_ref().map(|s| s.epoch).unwrap_or_default()
    }
}
