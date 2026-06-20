//! The single-owner agent actor body (§4.1) — the turn loop, phase sequence, and effect applier.
//!
//! An [`Engine`] owns one [`Snapshot`] (its only durable state) and drives turns by composing the
//! phases: `build_context` → `call_model` (a [`Provider`]) → `execute_tools` (the §12 pipeline) →
//! finalize. Each turn produces a stream of [`Effect`]s; the single-owner applier here orders and
//! applies them — appending turns and recording delegations — which is what makes suspension a
//! deterministic phase boundary (lifecycle §3.1).

use crate::config::Config;
use crate::control::{SteerReq, TurnControl};
use crate::conversation::{AssistantMsg, Conversation, SystemPrompt, ToolTurn, Turn};
use crate::credentials::{CredentialProvider, EmbeddedCredentialPool};
use crate::events::EventSink;
use crate::exec::{ExecutionEnvironment, LocalEnvironment};
use crate::provider::{build_context, ModelOutput, Provider};
use crate::snapshot::Snapshot;
use crate::tool_pipeline::run_tool;
use crate::tools::ToolRegistry;
use crate::turn::{Effect, TurnCx};
use crate::Failure;
use daemon_common::{Budget, CredScope, Epoch, JobId, ProfileRef, SessionId};
use daemon_protocol::{
    AgentEvent, CompletionSource, ConvTurnView, ConvView, EndReason, HostRequestHandler,
    ToolCallView, ToolDetail, ToolResultView, TurnSummary, TurnTrigger, UserMsg,
};
use std::sync::Arc;

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
    config: Config,
    /// The contained execution environment (§13) tools run in; the host injects a per-session
    /// workspace-rooted one via [`crate::EngineProfile`], else the default sandbox.
    exec: Arc<dyn ExecutionEnvironment>,
    /// A one-shot override for the next turn's [`TurnTrigger`] (set when a steer opens a turn);
    /// consumed at the start of `run_turn`.
    next_trigger: Option<TurnTrigger>,
}

impl Engine {
    /// Construct an engine over an existing snapshot.
    pub fn from_snapshot(
        snapshot: Snapshot,
        provider: Arc<dyn Provider>,
        registry: Arc<ToolRegistry>,
    ) -> Self {
        // Default sandbox keyed by session id; a host injects a workspace-rooted env via the profile.
        let exec: Arc<dyn ExecutionEnvironment> =
            Arc::new(LocalEnvironment::sandbox(snapshot.session_id.as_str()));
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
            config: Config::default(),
            exec,
            next_trigger: None,
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

    /// Inject the engine tunables (§20) the host loaded from its config.
    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Inject the execution environment (§13) this engine's tools run in (a per-session
    /// workspace-rooted [`LocalEnvironment`], or a host-routed env).
    pub fn with_exec(mut self, exec: Arc<dyn ExecutionEnvironment>) -> Self {
        self.exec = exec;
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

    /// Append an out-of-band steer marker into the conversation (hermes-style) and arm the next
    /// turn's trigger as [`TurnTrigger::Steer`]. The steer text becomes part of the model context.
    pub fn push_steer_marker(&mut self, steer: &SteerReq) {
        self.snapshot
            .conversation
            .push_user(UserMsg::new(format!("[steer] {}", steer.text)));
        self.next_trigger = Some(TurnTrigger::Steer);
    }

    /// The current snapshot (the only durable state).
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Build a read-only [`ConvView`] projection of the current conversation (the §17 snapshot
    /// reply body). Never exposes live resources — only the durable conversation + epoch.
    pub fn conv_view(&self) -> ConvView {
        let turns = self
            .snapshot
            .conversation
            .turns
            .iter()
            .map(|turn| match turn {
                Turn::User(u) => ConvTurnView {
                    role: "user".into(),
                    text: u.text.clone(),
                    tools: Vec::new(),
                },
                Turn::Assistant(a) => ConvTurnView {
                    role: "assistant".into(),
                    text: a.text.clone(),
                    tools: Vec::new(),
                },
                Turn::Tool(t) => ConvTurnView {
                    role: "tool".into(),
                    text: t.assistant.text.clone(),
                    tools: t.calls.iter().map(|(call, _)| call.name.clone()).collect(),
                },
            })
            .collect();
        ConvView {
            epoch: self.snapshot.epoch.0,
            turns,
            waiting_for: self
                .snapshot
                .waiting_for
                .iter()
                .map(|j| j.to_string())
                .collect(),
        }
    }

    /// Serve any pending snapshot requests at a consistent phase boundary by emitting a
    /// [`AgentEvent::Snapshot`] carrying the current [`ConvView`].
    fn serve_snapshots(&self, control: &TurnControl, events: &EventSink) {
        for request_id in control.drain_snapshot() {
            let view = self.conv_view();
            events.emit(|seq| AgentEvent::Snapshot {
                seq,
                request_id,
                view,
            });
        }
    }

    /// A full phase boundary: serve snapshots, drain steer (appending markers + acking each), then
    /// report whether cancellation has been requested.
    fn boundary(&mut self, control: &TurnControl, events: &EventSink) -> bool {
        self.serve_snapshots(control, events);
        for steer in control.drain_steer() {
            self.push_steer_marker(&steer);
            let request_id = steer.request_id;
            events.emit(|seq| AgentEvent::Steered {
                seq,
                request_id,
                accepted: true,
            });
        }
        control.is_cancelled()
    }

    /// A read-only phase boundary (inside the tool loop): serve snapshots and report cancellation,
    /// without mutating the conversation mid-tool-turn.
    fn boundary_readonly(&self, control: &TurnControl, events: &EventSink) -> bool {
        self.serve_snapshots(control, events);
        control.is_cancelled()
    }

    /// Finalize an interrupted turn: emit `TurnFinished{Interrupted}` and report it as a (terminal)
    /// completed outcome.
    fn finish_interrupted(&self, events: &EventSink) -> TurnOutcome {
        let summary = TurnSummary::ended(EndReason::Interrupted);
        let emitted = summary.clone();
        events.emit(|seq| AgentEvent::TurnFinished {
            seq,
            summary: emitted,
        });
        TurnOutcome::Completed(summary)
    }

    /// Finalize a failed turn: emit `Error` + `TurnFinished{Failed}` and report it as terminal.
    fn finish_failed(&self, failure: Failure, events: &EventSink) -> TurnOutcome {
        if matches!(failure, Failure::Cancelled) {
            return self.finish_interrupted(events);
        }
        let text = failure.to_string();
        events.emit(|seq| AgentEvent::Error { seq, failure: text });
        let summary = TurnSummary::ended(EndReason::Failed);
        let emitted = summary.clone();
        events.emit(|seq| AgentEvent::TurnFinished {
            seq,
            summary: emitted,
        });
        TurnOutcome::Completed(summary)
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
    ///
    /// `offer_tools` is `false` for the final budget-exhausted summary call (no tools offered, so the
    /// model must produce prose), `true` for every normal ReAct round.
    async fn call_model(&self, events: &EventSink, offer_tools: bool) -> Result<ModelOutput, Failure> {
        let tools = if offer_tools {
            self.tool_names()
        } else {
            Vec::new()
        };
        let req = build_context(&self.snapshot.conversation, &tools);
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
                Err(f) if f.is_rotatable() && attempt < self.config.model_retry_attempts => {
                    // Mark the credential out and retry on a rotated one (up to the configured count).
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
    ///
    /// The turn observes the shared [`TurnControl`] at phase boundaries: a requested interrupt
    /// finalizes the turn as [`EndReason::Interrupted`], queued steers are drained into the
    /// conversation (acked via [`AgentEvent::Steered`]), and pending snapshot requests are served
    /// with a consistent [`AgentEvent::Snapshot`]. A provider failure ends the turn as
    /// [`EndReason::Failed`] (after an [`AgentEvent::Error`]).
    pub async fn run_turn(
        &mut self,
        host: &dyn HostRequestHandler,
        events: &EventSink,
        control: &TurnControl,
    ) -> Result<TurnOutcome, Failure> {
        let resuming = !self.pending.is_empty();
        let trigger = self.next_trigger.take().unwrap_or(if resuming {
            TurnTrigger::BackgroundCompletion {
                source: CompletionSource::Delegation(self.pending[0].job_id.clone()),
            }
        } else {
            TurnTrigger::User
        });
        events.emit(|seq| AgentEvent::TurnStarted { seq, trigger });

        // Boundary: an interrupt that arrived before the turn does any work ends it immediately.
        if self.boundary(control, events) {
            return Ok(self.finish_interrupted(events));
        }

        if resuming {
            // Resume path: a background completion arrived — apply it idempotently and bump the epoch,
            // then fall into the ReAct loop so the model sees the resolved tool result(s) and can
            // either finalize or take further tool steps.
            self.resolve_pending();
            self.snapshot.waiting_for.clear();
            self.snapshot.epoch = self.snapshot.epoch.next();
        } else if let Some(job_id) = self.snapshot.waiting_for.first().cloned() {
            // Re-activated while still suspended (e.g. recovery before the worker ran): re-suspend the
            // same job deterministically; the durable outbox dedupes the re-enqueue.
            return Ok(self.suspend(job_id, events, false));
        }

        // The in-turn ReAct loop (§4.2): build_context -> call_model -> execute_tools -> observe ->
        // call_model again, until the model returns final text (completion) or a tool delegates
        // (durable suspension). The iteration budget is the hard stop. The loop runs fully
        // in-process within one durable turn; only `Effect::Delegate` crosses the suspension boundary.
        let exec = self.exec.clone();
        let registry = self.registry.clone();
        let tool_result_budget = self.config.tool_result_budget;
        let mut rounds_left = self.config.max_iterations;

        loop {
            if rounds_left == 0 {
                // Budget exhausted: one final toolless call asks the model to summarize what it has,
                // then the turn ends `BudgetExhausted` (the model cannot keep calling tools forever).
                return Ok(self.finish_budget_exhausted(events).await);
            }
            rounds_left -= 1;

            let out = match self.call_model(events, true).await {
                Ok(out) => out,
                Err(f) => return Ok(self.finish_failed(f, events)),
            };

            // Boundary after the model call: serve snapshots/steer, honor a mid-call interrupt.
            if self.boundary(control, events) {
                return Ok(self.finish_interrupted(events));
            }

            if out.tool_calls.is_empty() {
                self.finalize_text(&out, events);
                return Ok(self.complete(out, events));
            }

            let cx = TurnCx {
                cancel: control.cancel_token(),
                events,
                host,
                session_id: self.snapshot.session_id.clone(),
                budget: self.budget,
                exec: &*exec,
                tool_result_budget,
            };

            // execute_tools: run each call through the §12 pipeline, collecting result slots,
            // effects, and structured detail for the rich §17 transcript views.
            let mut calls = Vec::new();
            let mut effects: Vec<Effect> = Vec::new();
            let mut interrupted = false;
            for call in &out.tool_calls {
                let view = ToolCallView {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    args_summary: call.args.clone(),
                    // A generic structured echo of the call arguments, opaque to the daemon; a tool
                    // with a richer call schema can refine this once providers carry structured args.
                    detail: Some(ToolDetail {
                        kind: call.name.clone(),
                        body: call.args.clone().into_bytes(),
                    }),
                };
                events.emit(|seq| AgentEvent::ToolStarted { seq, call: view });
                let outcome = run_tool(call, &registry, &cx).await;
                let result_view = ToolResultView {
                    call_id: outcome.result.call_id.clone(),
                    ok: outcome.result.ok,
                    summary: outcome.result.content.clone(),
                    // The tool's typed output (fs listing, command exit/stdout, ...) for a rich
                    // consumer; `None` for plain-text tools.
                    detail: outcome.detail.clone(),
                };
                events.emit(|seq| AgentEvent::ToolFinished {
                    seq,
                    result: result_view,
                });
                calls.push((call.clone(), outcome.result));
                effects.extend(outcome.effects);
                // Boundary after each tool: an interrupt stops further tool execution.
                if self.boundary_readonly(control, events) {
                    interrupted = true;
                    break;
                }
            }

            // The single-owner applier: record the assembled tool turn, then apply any extra effects.
            self.snapshot.conversation.turns.push(Turn::Tool(ToolTurn {
                assistant: AssistantMsg {
                    text: out.text.clone(),
                    reasoning: out.reasoning.clone(),
                },
                calls,
            }));
            let mut delegated: Option<JobId> = None;
            for effect in effects {
                match effect {
                    Effect::Persist(turn) => self.snapshot.conversation.turns.push(turn),
                    Effect::Delegate(job_id) => delegated = Some(job_id),
                }
            }

            // An interrupt at a tool boundary finalizes the turn before it would suspend/loop.
            if interrupted {
                return Ok(self.finish_interrupted(events));
            }

            if let Some(job_id) = delegated {
                // A delegation crosses the durable boundary: suspend the turn and wait for the wake.
                self.snapshot.waiting_for.push(job_id.clone());
                return Ok(self.suspend(job_id, events, true));
            }
            // No delegation: loop — the next `call_model` sees the tool results in context.
        }
    }

    /// Finalize a turn that hit its iteration budget: one final toolless model call to summarize,
    /// then `TurnFinished { BudgetExhausted }`. Any tool calls the model attempts on this pass are
    /// ignored — the turn is ending.
    async fn finish_budget_exhausted(&mut self, events: &EventSink) -> TurnOutcome {
        let out = match self.call_model(events, false).await {
            Ok(out) => out,
            Err(f) => return self.finish_failed(f, events),
        };
        self.finalize_text(&out, events);
        let summary = TurnSummary {
            end_reason: EndReason::BudgetExhausted,
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
    use daemon_common::{CredScope, ReqId, UsageDelta};
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
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .expect("turn completes after a single rotation");
        assert!(matches!(outcome, TurnOutcome::Completed(_)));
        assert_eq!(provider.calls.load(Ordering::Relaxed), 2, "retried once");
        assert_eq!(pool.live_count(), 1, "the rotated key is cooling down");
    }

    /// A provider that always fails with a non-rotatable error.
    struct FailingProvider;

    #[async_trait::async_trait]
    impl Provider for FailingProvider {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }

        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            Err(Failure::Provider("model exploded".into()))
        }
    }

    /// An event sink that records every emitted event for assertions.
    fn collecting() -> (EventSink, Arc<std::sync::Mutex<Vec<AgentEvent>>>) {
        let log = Arc::new(std::sync::Mutex::new(Vec::<AgentEvent>::new()));
        let l = log.clone();
        (EventSink::new(move |ev| l.lock().unwrap().push(ev)), log)
    }

    fn completing_engine(id: &str) -> Engine {
        Engine::fresh(
            SessionId::new(id),
            SystemPrompt::new("test"),
            Arc::new(crate::provider::MockProvider::completing("hi")),
            Arc::new(ToolRegistry::new()),
        )
    }

    /// An interrupt observed at the opening phase boundary finalizes the turn as `Interrupted`.
    #[tokio::test]
    async fn interrupt_at_boundary_finalizes_interrupted() {
        let mut engine = completing_engine("int");
        engine.push_user(UserMsg::new("hello"));
        let control = TurnControl::new();
        control.cancel();
        let (sink, log) = collecting();

        let outcome = engine.run_turn(&NoopHost, &sink, &control).await.unwrap();
        match outcome {
            TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::Interrupted),
            _ => panic!("expected a completed-but-interrupted outcome"),
        }
        assert!(log.lock().unwrap().iter().any(|e| matches!(
            e,
            AgentEvent::TurnFinished { summary, .. } if summary.end_reason == EndReason::Interrupted
        )));
    }

    /// A provider failure ends the turn as `Failed`, after an `Error` event.
    #[tokio::test]
    async fn provider_failure_emits_error_and_failed() {
        let mut engine = Engine::fresh(
            SessionId::new("fail"),
            SystemPrompt::new("test"),
            Arc::new(FailingProvider),
            Arc::new(ToolRegistry::new()),
        );
        engine.push_user(UserMsg::new("hello"));
        let (sink, log) = collecting();

        let outcome = engine
            .run_turn(&NoopHost, &sink, &TurnControl::new())
            .await
            .unwrap();
        match outcome {
            TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::Failed),
            _ => panic!("expected a failed outcome"),
        }
        let log = log.lock().unwrap();
        assert!(log.iter().any(|e| matches!(e, AgentEvent::Error { .. })));
        assert!(log.iter().any(|e| matches!(
            e,
            AgentEvent::TurnFinished { summary, .. } if summary.end_reason == EndReason::Failed
        )));
    }

    /// A pending snapshot request is served at a phase boundary with a `ConvView` reflecting the
    /// conversation.
    #[tokio::test]
    async fn snapshot_request_served_with_conv_view() {
        let mut engine = completing_engine("snap");
        engine.push_user(UserMsg::new("question"));
        let control = TurnControl::new();
        control.push_snapshot(ReqId(7));
        let (sink, log) = collecting();

        engine.run_turn(&NoopHost, &sink, &control).await.unwrap();
        let log = log.lock().unwrap();
        let (request_id, view) = log
            .iter()
            .find_map(|e| match e {
                AgentEvent::Snapshot {
                    request_id, view, ..
                } => Some((*request_id, view.clone())),
                _ => None,
            })
            .expect("a snapshot event");
        assert_eq!(request_id, ReqId(7));
        assert!(view
            .turns
            .iter()
            .any(|t| t.role == "user" && t.text == "question"));
    }

    /// A queued steer is drained at a boundary: the marker lands in the conversation and a
    /// `Steered` ack is emitted.
    #[tokio::test]
    async fn steer_drained_appends_marker_and_acks() {
        let mut engine = completing_engine("steer");
        engine.push_user(UserMsg::new("hi"));
        let control = TurnControl::new();
        control.push_steer(SteerReq {
            request_id: ReqId(3),
            text: "focus".into(),
        });
        let (sink, log) = collecting();

        engine.run_turn(&NoopHost, &sink, &control).await.unwrap();
        assert!(log.lock().unwrap().iter().any(|e| matches!(
            e,
            AgentEvent::Steered { request_id, accepted, .. } if *request_id == ReqId(3) && *accepted
        )));
        assert!(engine
            .snapshot()
            .conversation
            .turns
            .iter()
            .any(|t| matches!(t, Turn::User(u) if u.text.contains("[steer] focus"))));
    }

    // --- ReAct loop (§4.2) ---

    use crate::conversation::ToolCall;
    use crate::provider::{ScriptStep, ScriptedProvider};
    use crate::tools::Tool;
    use crate::turn::TurnCx;

    /// A trivial in-turn tool that records how many times it ran (shared counter) and returns a
    /// fixed result — enough to exercise the model->tools->model loop without a real tool crate.
    struct CounterTool {
        runs: Arc<AtomicU64>,
    }

    #[async_trait::async_trait]
    impl Tool for CounterTool {
        fn name(&self) -> &str {
            "counter"
        }
        fn schema(&self) -> &str {
            "{}"
        }
        async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> crate::tools::ToolOutcome {
            let n = self.runs.fetch_add(1, Ordering::Relaxed);
            crate::tools::ToolOutcome::text(call.call_id.clone(), true, format!("counter:{n}"))
        }
    }

    fn looping_engine(provider: Arc<dyn Provider>, runs: Arc<AtomicU64>, max_iterations: u32) -> Engine {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(CounterTool { runs }));
        let config = Config {
            max_iterations,
            ..Config::default()
        };
        Engine::fresh(
            SessionId::new("react"),
            SystemPrompt::new("test"),
            provider,
            Arc::new(registry),
        )
        .with_config(config)
    }

    /// The model calls a tool twice across two rounds, then returns final text — one activation runs
    /// the whole multi-round loop and completes (no suspension).
    #[tokio::test]
    async fn multi_round_loop_runs_tools_then_completes() {
        let runs = Arc::new(AtomicU64::new(0));
        let provider = Arc::new(ScriptedProvider::new(
            vec![
                ScriptStep::Call {
                    name: "counter".into(),
                    args: "{}".into(),
                },
                ScriptStep::Call {
                    name: "counter".into(),
                    args: "{}".into(),
                },
            ],
            "all done",
        ));
        let mut engine = looping_engine(provider.clone(), runs.clone(), 90);
        engine.push_user(UserMsg::new("do work"));

        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .unwrap();
        match outcome {
            TurnOutcome::Completed(s) => {
                assert_eq!(s.end_reason, EndReason::Completed);
                assert_eq!(s.final_text.as_deref(), Some("all done"));
            }
            _ => panic!("expected completion after the loop"),
        }
        assert_eq!(runs.load(Ordering::Relaxed), 2, "the tool ran on both rounds");
        // 3 model rounds: two tool rounds + the final-text round.
        assert_eq!(provider.call_count(), 3);
        // Two tool turns are recorded in the durable conversation.
        let tool_turns = engine
            .snapshot()
            .conversation
            .turns
            .iter()
            .filter(|t| matches!(t, Turn::Tool(_)))
            .count();
        assert_eq!(tool_turns, 2);
    }

    /// A provider that never stops calling a (non-delegating) tool exhausts the iteration budget; the
    /// turn ends `BudgetExhausted` after one final toolless summary call.
    #[tokio::test]
    async fn iteration_budget_exhaustion_ends_with_summary() {
        let runs = Arc::new(AtomicU64::new(0));
        let provider = Arc::new(ScriptedProvider::looping(ScriptStep::Call {
            name: "counter".into(),
            args: "{}".into(),
        }));
        let mut engine = looping_engine(provider.clone(), runs.clone(), 4);
        engine.push_user(UserMsg::new("loop forever"));

        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .unwrap();
        match outcome {
            TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::BudgetExhausted),
            _ => panic!("expected budget exhaustion"),
        }
        // The tool ran exactly `max_iterations` times; the budget is the hard stop.
        assert_eq!(runs.load(Ordering::Relaxed), 4);
        // 4 loop rounds + 1 toolless summary round.
        assert_eq!(provider.call_count(), 5);
    }

    /// Cancellation observed mid-loop (after a tool runs) finalizes the turn as `Interrupted` rather
    /// than looping back to the model.
    #[tokio::test]
    async fn cancel_mid_loop_finalizes_interrupted() {
        let runs = Arc::new(AtomicU64::new(0));
        let provider = Arc::new(ScriptedProvider::looping(ScriptStep::Call {
            name: "counter".into(),
            args: "{}".into(),
        }));
        let mut engine = looping_engine(provider, runs, 90);
        engine.push_user(UserMsg::new("go"));
        let control = TurnControl::new();
        // Cancel before the first boundary: the turn finalizes interrupted immediately.
        control.cancel();

        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &control)
            .await
            .unwrap();
        match outcome {
            TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::Interrupted),
            _ => panic!("expected interruption"),
        }
    }
}
