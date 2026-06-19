//! The single-owner agent actor body (§4.1) — the turn loop, phase sequence, and effect applier.
//!
//! An [`Engine`] owns one [`Snapshot`] (its only durable state) and drives turns by composing the
//! phases: `build_context` → `call_model` (a [`Provider`]) → `execute_tools` (the §12 pipeline) →
//! finalize. Each turn produces a stream of [`Effect`]s; the single-owner applier here orders and
//! applies them — appending turns and recording delegations — which is what makes suspension a
//! deterministic phase boundary (lifecycle §3.1).

use crate::conversation::{AssistantMsg, Conversation, SystemPrompt, ToolTurn, Turn};
use crate::credentials::{CredentialProvider, EmbeddedCredentialPool};
use crate::events::EventSink;
use crate::provider::{build_context, ModelOutput, Provider};
use crate::snapshot::Snapshot;
use crate::tool_pipeline::run_tool;
use crate::tools::ToolRegistry;
use crate::turn::{Effect, TurnCx};
use crate::Failure;
use daemon_common::{Budget, CredScope, Epoch, JobId, ProfileRef, SessionId};
use daemon_protocol::{
    AgentEvent, CompletionSource, EndReason, HostRequestHandler, ToolCallView, ToolResultView,
    TurnSummary, TurnTrigger, UserMsg,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// A background-job completion handed back to the engine on rehydration (the core-local form of the
/// durable `JobCompletion`; the host adapter converts between them).
#[derive(Clone, Debug)]
pub struct Completion {
    /// The job that completed.
    pub job_id: JobId,
    /// The completion payload.
    pub payload: Vec<u8>,
}

/// What one turn resolved to.
pub enum TurnOutcome {
    /// The turn reached a terminal state.
    Completed(TurnSummary),
    /// The turn suspended at a phase boundary, delegating background work.
    Suspended(Suspension),
}

/// The durable handoff produced when a turn suspends: the job to enqueue and the post-bump epoch.
pub struct Suspension {
    /// The delegated job.
    pub job_id: JobId,
    /// The epoch the snapshot now carries (bumped at this suspension boundary).
    pub epoch: Epoch,
    /// The opaque work payload for the background worker.
    pub payload: Vec<u8>,
}

/// The single-owner agent engine (§4.1).
pub struct Engine {
    snapshot: Snapshot,
    provider: Arc<dyn Provider>,
    registry: Arc<ToolRegistry>,
    pending: Vec<Completion>,
    budget: Budget,
    credentials: Arc<dyn CredentialProvider>,
    profile: ProfileRef,
}

impl Engine {
    /// Construct an engine over an existing snapshot.
    pub fn from_snapshot(
        snapshot: Snapshot,
        provider: Arc<dyn Provider>,
        registry: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            snapshot,
            provider,
            registry,
            pending: Vec::new(),
            budget: Budget::unlimited(),
            // L1 default: the in-tree embedded pool. Under a host an authority-backed provider is
            // injected via `with_credentials` (host-spec §6).
            credentials: Arc::new(EmbeddedCredentialPool::single_key()),
            profile: ProfileRef::new("default"),
        }
    }

    /// Construct a fresh engine for a new session with the given system prompt.
    pub fn fresh(
        session_id: SessionId,
        system: SystemPrompt,
        provider: Arc<dyn Provider>,
        registry: Arc<ToolRegistry>,
    ) -> Self {
        let mut snapshot = Snapshot::fresh(session_id);
        snapshot.conversation = Conversation::new(system);
        Self::from_snapshot(snapshot, provider, registry)
    }

    /// Set the budget governing this engine's turns.
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Inject the credential provider + profile this engine acquires capabilities from (the host
    /// injects an authority-backed or brokered impl; the default is the embedded L1 pool).
    pub fn with_credentials(
        mut self,
        credentials: Arc<dyn CredentialProvider>,
        profile: ProfileRef,
    ) -> Self {
        self.credentials = credentials;
        self.profile = profile;
        self
    }

    /// Stash background-job completions to be applied (idempotently) before the next turn runs.
    pub fn apply_completions(&mut self, completions: Vec<Completion>) {
        self.pending.extend(completions);
    }

    /// Append a user message that opens the next turn.
    pub fn push_user(&mut self, input: UserMsg) {
        self.snapshot.conversation.push_user(input);
    }

    /// The current snapshot (the only durable state).
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// The current incarnation epoch.
    pub fn epoch(&self) -> Epoch {
        self.snapshot.epoch
    }

    /// The registered tool names (offered to the model each turn).
    fn tool_names(&self) -> Vec<String> {
        self.registry.names()
    }

    /// `call_model` phase: acquire a scoped credential capability, flatten context, ask the
    /// provider under that capability, stream usage + reasoning. On a rotatable failure
    /// (quota/auth) the credential is rotated and the call retried once (§7 `should_rotate`).
    async fn call_model(&self, events: &EventSink) -> Result<ModelOutput, Failure> {
        let req = build_context(&self.snapshot.conversation, &self.tool_names());
        // The scope a turn needs: the `chat` action on this engine's profile.
        let want = CredScope::new([self.profile.as_str()], ["chat"], self.budget.tokens);
        let mut attempt = 0u8;
        let out = loop {
            let lease = self
                .credentials
                .acquire(&self.profile, &want)
                .await
                .map_err(|e| Failure::Provider(format!("credential acquire: {e}")))?;
            let result = self.provider.chat(req.clone()).await;
            self.credentials.release(&lease).await;
            match result {
                Ok(out) => break out,
                Err(f) if f.is_rotatable() && attempt == 0 => {
                    // Mark the credential out and retry on a rotated one.
                    self.credentials.rotate(&self.profile, &lease.cap_id).await;
                    attempt += 1;
                    continue;
                }
                Err(f) => return Err(f),
            }
        };
        let usage = out.usage;
        events.emit(|seq| AgentEvent::Usage { seq, delta: usage });
        if let Some(reasoning) = &out.reasoning {
            let reasoning = reasoning.clone();
            events.emit(|seq| AgentEvent::ReasoningDelta {
                seq,
                text: reasoning,
            });
        }
        Ok(out)
    }

    /// Finalize a text-only turn: stream the text and append the assistant turn.
    fn finalize_text(&mut self, out: &ModelOutput, events: &EventSink) {
        if !out.text.is_empty() {
            let text = out.text.clone();
            events.emit(|seq| AgentEvent::TextDelta { seq, text });
        }
        self.snapshot.conversation.push_assistant(AssistantMsg {
            text: out.text.clone(),
            reasoning: out.reasoning.clone(),
        });
    }

    /// Idempotently resolve pending completions into the conversation's open tool result slots.
    fn resolve_pending(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        for completion in pending {
            let payload = String::from_utf8_lossy(&completion.payload).to_string();
            for turn in self.snapshot.conversation.turns.iter_mut() {
                if let Turn::Tool(tool_turn) = turn {
                    for (_call, result) in tool_turn.calls.iter_mut() {
                        if result.content.contains(completion.job_id.as_str()) {
                            // Deterministic value => applying the same completion twice is a no-op.
                            result.ok = true;
                            result.content = format!("completed:{}:{}", completion.job_id, payload);
                        }
                    }
                }
            }
        }
    }

    /// Run one turn to a phase boundary: terminal completion or durable suspension (§4.2 / §3.1).
    pub async fn run_turn(
        &mut self,
        host: &dyn HostRequestHandler,
        events: &EventSink,
        cancel: CancellationToken,
    ) -> Result<TurnOutcome, Failure> {
        let resuming = !self.pending.is_empty();
        let trigger = if resuming {
            TurnTrigger::BackgroundCompletion {
                source: CompletionSource::Delegation(self.pending[0].job_id.clone()),
            }
        } else {
            TurnTrigger::User
        };
        events.emit(|seq| AgentEvent::TurnStarted { seq, trigger });

        // Resume path: a background completion arrived — apply it idempotently, bump the epoch, and
        // let the model finalize.
        if resuming {
            self.resolve_pending();
            self.snapshot.waiting_for.clear();
            self.snapshot.epoch = self.snapshot.epoch.next();
            let out = self.call_model(events).await?;
            self.finalize_text(&out, events);
            return Ok(self.complete(out, events));
        }

        // Re-activated while still suspended (e.g. recovery before the worker ran): re-suspend the
        // same job deterministically; the durable outbox dedupes the re-enqueue.
        if let Some(job_id) = self.snapshot.waiting_for.first().cloned() {
            return Ok(self.suspend(job_id, events, false));
        }

        // Fresh activation: ask the model. A completing provider finishes here; a delegating one
        // returns a tool call that drives suspension.
        let out = self.call_model(events).await?;
        if out.tool_calls.is_empty() {
            self.finalize_text(&out, events);
            return Ok(self.complete(out, events));
        }

        let cx = TurnCx {
            cancel,
            events,
            host,
            session_id: self.snapshot.session_id.clone(),
            budget: self.budget,
        };
        let registry = self.registry.clone();

        // execute_tools: run each call through the §12 pipeline, collecting result slots + effects.
        let mut calls = Vec::new();
        let mut effects: Vec<Effect> = Vec::new();
        for call in &out.tool_calls {
            let view = ToolCallView {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                args_summary: call.args.clone(),
            };
            events.emit(|seq| AgentEvent::ToolStarted { seq, call: view });
            let outcome = run_tool(call, &registry, &cx).await;
            let result_view = ToolResultView {
                call_id: outcome.result.call_id.clone(),
                ok: outcome.result.ok,
                summary: outcome.result.content.clone(),
            };
            events.emit(|seq| AgentEvent::ToolFinished {
                seq,
                result: result_view,
            });
            calls.push((call.clone(), outcome.result));
            effects.extend(outcome.effects);
        }

        // The single-owner applier: the assembled tool turn is the leading Persist effect, then any
        // Delegate effects, applied in order.
        effects.insert(
            0,
            Effect::Persist(Turn::Tool(ToolTurn {
                assistant: AssistantMsg {
                    text: out.text.clone(),
                    reasoning: out.reasoning.clone(),
                },
                calls,
            })),
        );
        let mut delegated: Option<JobId> = None;
        for effect in effects {
            match effect {
                Effect::Persist(turn) => self.snapshot.conversation.turns.push(turn),
                Effect::Delegate(job_id) => delegated = Some(job_id),
            }
        }

        match delegated {
            Some(job_id) => {
                self.snapshot.waiting_for.push(job_id.clone());
                Ok(self.suspend(job_id, events, true))
            }
            None => Ok(self.complete(out, events)),
        }
    }

    /// Emit the terminal `TurnFinished` and build the completed outcome.
    fn complete(&self, out: ModelOutput, events: &EventSink) -> TurnOutcome {
        let summary = TurnSummary {
            end_reason: EndReason::Completed,
            final_text: Some(out.text),
            usage: out.usage,
        };
        let emitted = summary.clone();
        events.emit(|seq| AgentEvent::TurnFinished {
            seq,
            summary: emitted,
        });
        TurnOutcome::Completed(summary)
    }

    /// Emit the suspending `TurnFinished` and build the suspension handoff, bumping the epoch on a
    /// fresh suspension (but not on a deterministic recovery re-suspend).
    fn suspend(&mut self, job_id: JobId, events: &EventSink, bump_epoch: bool) -> TurnOutcome {
        if bump_epoch {
            self.snapshot.epoch = self.snapshot.epoch.next();
        }
        let summary = TurnSummary::ended(EndReason::Suspended);
        events.emit(|seq| AgentEvent::TurnFinished { seq, summary });
        TurnOutcome::Suspended(Suspension {
            job_id,
            epoch: self.snapshot.epoch,
            payload: b"delegated-work".to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Capabilities, ModelOutput, Request, ToolCallFormat};
    use daemon_common::{CredScope, UsageDelta};
    use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A provider that fails the first call with a rotatable error, then completes.
    struct FlakyProvider {
        calls: AtomicU64,
    }

    #[async_trait::async_trait]
    impl Provider for FlakyProvider {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }

        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed);
            if n == 0 {
                Err(Failure::Rotatable("quota exceeded (429)".into()))
            } else {
                Ok(ModelOutput {
                    text: "done after rotation".into(),
                    reasoning: None,
                    tool_calls: Vec::new(),
                    usage: UsageDelta::default(),
                })
            }
        }
    }

    struct NoopHost;

    #[async_trait::async_trait]
    impl HostRequestHandler for NoopHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved(true),
            }
        }
    }

    /// A rotatable provider failure rotates the credential and retries on a second key — the turn
    /// completes, the provider was called twice, and one key is now cooling down.
    #[tokio::test]
    async fn rotatable_failure_rotates_credential_and_retries() {
        let provider = Arc::new(FlakyProvider {
            calls: AtomicU64::new(0),
        });
        let pool = Arc::new(EmbeddedCredentialPool::new(
            "openai",
            CredScope::new(["openai"], ["chat"], None),
            [
                ("key-a".to_string(), "secret-a".to_string()),
                ("key-b".to_string(), "secret-b".to_string()),
            ],
            60_000,
            30_000,
        ));
        let mut engine = Engine::fresh(
            SessionId::new("rotate"),
            SystemPrompt::new("test"),
            provider.clone(),
            Arc::new(ToolRegistry::new()),
        )
        .with_credentials(pool.clone(), ProfileRef::new("openai"));
        engine.push_user(UserMsg::new("hello"));

        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), CancellationToken::new())
            .await
            .expect("turn completes after a single rotation");
        assert!(matches!(outcome, TurnOutcome::Completed(_)));
        assert_eq!(provider.calls.load(Ordering::Relaxed), 2, "retried once");
        assert_eq!(pool.live_count(), 1, "the rotated key is cooling down");
    }
}
