//! The single-owner agent actor body (§4.1) — the turn loop, phase sequence, and effect applier.
//!
//! An [`Engine`] owns one [`Snapshot`] (its only durable state) and drives turns by composing the
//! phases: `build_context` → `call_model` (a [`Provider`]) → `execute_tools` (the §12 pipeline) →
//! finalize. Each turn produces a stream of [`Effect`]s; the single-owner applier here orders and
//! applies them — appending turns and recording delegations — which is what makes suspension a
//! deterministic phase boundary (lifecycle §3.1).

use crate::config::Config;
use crate::context::{BudgetedContextEngine, ContextEngine, PromptAssembler, StablePromptSource};
use crate::control::{SteerReq, TurnControl};
use crate::conversation::{AssistantMsg, Conversation, SystemPrompt, ToolTurn, Turn};
use crate::credentials::{CredentialProvider, EmbeddedCredentialPool};
use crate::events::EventSink;
use crate::exec::{ExecutionEnvironment, LocalEnvironment};
use crate::memory::{MemoryProvider, RecallQuery, SwitchReason};
use crate::provider::{ModelOutput, Provider};
use crate::recovery::{drive_model_call, ModelCallPolicy, RecoveryStep};
use crate::snapshot::Snapshot;
use crate::tool_pipeline::run_tool;
use crate::tools::ToolRegistry;
use crate::turn::{Effect, TurnCx};
use crate::Failure;
use daemon_common::{
    Budget, CredScope, Epoch, JobId, ProfileRef, RateLimitSnapshot, ReqId, SessionId, UsageDelta,
};
use daemon_protocol::{
    AgentEvent, CompletionSource, ContextStatus, ConvTurnView, ConvView, EndReason, HostRequest,
    HostRequestHandler, HostRequestKind, SpawnSeed, SpawnSpec, ToolCallView, ToolDetail,
    ToolResultView, TurnSummary, TurnTrigger, UserMsg,
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

/// The suspension payload marking an **approval park** (§12 HITL) rather than a delegation: the
/// host parks it for an operator decision instead of enqueuing a runnable background job.
pub const APPROVAL_SUSPEND_PAYLOAD: &[u8] = b"await-approval";

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
    /// The single fallback profile the §8 recovery loop hops to when the active profile cannot
    /// recover (persistent auth/billing/content-policy). `None` disables the hop (the default).
    fallback_profile: Option<ProfileRef>,
    config: Config,
    /// The contained execution environment (§13) tools run in; the host injects a per-session
    /// workspace-rooted one via [`crate::EngineProfile`], else the default sandbox.
    exec: Arc<dyn ExecutionEnvironment>,
    /// The context engine (§10): prompt assembly, budget pressure, and compaction. Defaults to the
    /// cheap [`BudgetedContextEngine`] (drop-oldest).
    context: Arc<dyn ContextEngine>,
    /// The tiered prompt assembler (§10) for the current turn; memory (§11) populates its non-body
    /// tiers at turn start, the call_model phase folds them into the request.
    assembler: PromptAssembler,
    /// The registered memory providers (§11). Empty by default — memory is opt-in; the engine drives
    /// their hook order (`recall -> prompt_block -> before_compact -> after_turn`) around each turn.
    memory: Vec<Arc<dyn MemoryProvider>>,
    /// Generic stable-tier prompt sources (§10), independent of memory — e.g. the skills index.
    /// Folded into `assembler.stable` each turn; expected to be cache-stable across a conversation.
    prompt_sources: Vec<Arc<dyn StablePromptSource>>,
    /// A one-shot override for the next turn's [`TurnTrigger`] (set when a steer opens a turn);
    /// consumed at the start of `run_turn`.
    next_trigger: Option<TurnTrigger>,
    /// Whether the once-per-incarnation §10/§11 lifecycle hooks (`on_model`, `on_session_start`,
    /// `on_session_switch(Start|Resume)`) have fired yet. Set on the first `run_turn`.
    lifecycle_started: bool,
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
            fallback_profile: None,
            config: Config::default(),
            exec,
            context: Arc::new(BudgetedContextEngine::default()),
            assembler: PromptAssembler::default(),
            memory: Vec::new(),
            prompt_sources: Vec::new(),
            next_trigger: None,
            lifecycle_started: false,
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

    /// Inject the context engine (§10) this engine assembles/compacts context with (the default is
    /// the cheap [`BudgetedContextEngine`]).
    pub fn with_context_engine(mut self, context: Arc<dyn ContextEngine>) -> Self {
        self.context = context;
        self
    }

    /// Register the memory providers (§11) this engine consults around each turn (default empty).
    pub fn with_memory(mut self, memory: Vec<Arc<dyn MemoryProvider>>) -> Self {
        self.memory = memory;
        self
    }

    /// Register generic stable-tier prompt sources (§10) folded into the system prompt each turn
    /// (e.g. the skills index). Independent of memory; expected to be cache-stable.
    pub fn with_prompt_sources(mut self, sources: Vec<Arc<dyn StablePromptSource>>) -> Self {
        self.prompt_sources = sources;
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

    /// Set the single fallback profile the §8 recovery loop hops to when the active profile cannot
    /// recover a model failure (persistent auth/billing/content-policy).
    pub fn with_fallback_profile(mut self, profile: ProfileRef) -> Self {
        self.fallback_profile = Some(profile);
        self
    }

    /// Stash background-job completions to be applied (idempotently) before the next turn runs.
    pub fn apply_completions(&mut self, completions: Vec<Completion>) {
        self.pending.extend(completions);
    }

    /// Swap the model provider this engine calls — a live, per-session model switch. Refreshes the
    /// §10 context-window denominator from the new provider's [`Capabilities`](crate::provider::Capabilities).
    /// Intended to take effect at a turn boundary (the actor applies it between turns) so an
    /// in-flight turn's prompt cache is never invalidated mid-conversation.
    pub fn set_provider(&mut self, provider: Arc<dyn Provider>) {
        self.provider = provider;
        let info = self.model_info();
        self.context.on_model(&info);
    }

    /// Set this session's edit-approval [`ApprovalPolicy`](crate::approval::ApprovalPolicy) (the §12
    /// session mode) in place — a live, per-session switch. Recorded on the durable
    /// [`Snapshot`](crate::Snapshot) so it survives suspend/rehydrate, and consulted by the next
    /// gated tool action (it does not affect an in-flight gate already decided this turn).
    pub fn set_approval_policy(&mut self, policy: crate::approval::ApprovalPolicy) {
        self.snapshot.approval_policy = Some(policy);
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

    /// The registered tool definitions (name + JSON-Schema) offered to the model each turn.
    fn tool_defs(&self) -> Vec<crate::tools::ToolDef> {
        self.registry.defs()
    }

    /// `call_model` phase (§4.2) wrapped in §8 recovery: acquire a scoped credential, thread its
    /// secret onto the request, and drive the provider **stream** under the stale-stream watchdog +
    /// cancel token (streaming `TextDelta`/`ReasoningDelta`/`Usage` to the host). A failure is
    /// classified and bounded by [`ModelCallPolicy::decide`]:
    ///
    /// - *retry* (rate-limit/transport/overload/format): emit `RateLimit`, sleep a jittered backoff
    ///   (honoring `Retry-After`), retry on the same profile;
    /// - *rotate* (auth/quota): mark the credential out, retry on a rotated one;
    /// - *compact* (context/payload overflow): compact the context once and retry (the §8 -> §10
    ///   tie-in; a no-op until a [`ContextEngine`](crate::context::ContextEngine) is wired);
    /// - *fallback*: hop once to the configured fallback profile;
    /// - *abort*: surface the failure (the turn ends `Failed`).
    ///
    /// `offer_tools` is `false` for the final budget-exhausted summary call (no tools offered, so the
    /// model must produce prose), `true` for every normal ReAct round.
    async fn call_model(
        &mut self,
        events: &EventSink,
        offer_tools: bool,
        cancel: &CancellationToken,
    ) -> Result<ModelOutput, Failure> {
        let policy = ModelCallPolicy::from_config(&self.config);
        let tools = if offer_tools {
            self.tool_defs()
        } else {
            Vec::new()
        };
        let mut attempt = 0u32;
        let mut compacted = false;
        loop {
            // Rebuilt each attempt: a compaction step rewrites the conversation in place. The §10
            // assembler folds memory/stable tiers into the system preamble (empty by default).
            let mut req = self.assembler.assemble(&self.snapshot.conversation, &tools);
            // The scope a turn needs: the `chat` action on this engine's profile.
            let want = CredScope::new([self.profile.as_str()], ["chat"], self.budget.tokens);
            let lease = self
                .credentials
                .acquire(&self.profile, &want)
                .await
                .map_err(|e| Failure::Provider(format!("credential acquire: {e}")))?;
            // Thread the lease secret as the request's bearer credential (the §7 credential layer):
            // the deterministic providers ignore it; a networked provider sends it as `Authorization`.
            req.auth = lease.secret.as_ref().map(|s| s.expose().to_string());
            let result =
                drive_model_call(&*self.provider, req, cancel, policy.watchdog, events).await;
            self.credentials.release(&lease).await;
            let failure = match result {
                Ok(out) => return Ok(out),
                Err(f) => f,
            };
            match policy.decide(&failure, attempt) {
                RecoveryStep::Retry { after } => {
                    if let Failure::RateLimit { retry_after, .. } = &failure {
                        let reset_ms = retry_after.map(|d| d.as_millis() as u64);
                        events.emit(|seq| AgentEvent::RateLimit {
                            seq,
                            snapshot: RateLimitSnapshot {
                                remaining: None,
                                limit: None,
                                reset_ms,
                            },
                        });
                    }
                    tokio::time::sleep(after).await;
                    attempt += 1;
                }
                RecoveryStep::Rotate => {
                    self.credentials.rotate(&self.profile, &lease.cap_id).await;
                    attempt += 1;
                }
                RecoveryStep::Compact => {
                    // Compact at most once; if it freed nothing the overflow is unrecoverable.
                    if !compacted && self.compact_context(events).await {
                        compacted = true;
                    } else {
                        return Err(failure);
                    }
                }
                RecoveryStep::Fallback => {
                    // A single hop to the configured fallback profile with a fresh retry budget.
                    match self.fallback_profile.clone() {
                        Some(fb) if fb != self.profile => {
                            self.profile = fb;
                            attempt = 0;
                        }
                        _ => return Err(failure),
                    }
                }
                RecoveryStep::Abort => return Err(failure),
            }
        }
    }

    /// §11 `recall` + `prompt_block` gathering into the §10 assembler tiers, in spec order: each
    /// provider's recall results land in the `recalled` tier and its persistent `prompt_block` in
    /// the `stable` tier. A no-op when no [`MemoryProvider`](crate::memory::MemoryProvider) is
    /// registered. The recall query is the latest user message.
    async fn gather_memory(&mut self) {
        if self.memory.is_empty() {
            return;
        }
        let query = RecallQuery {
            text: self.latest_user_text(),
            top_k: 5,
        };
        for provider in &self.memory {
            if let Some(block) = provider.recall(&query).await {
                self.assembler.recalled.push(block.text);
            }
            if let Some(block) = provider.prompt_block() {
                self.assembler.stable.push(block.text);
            }
        }
    }

    /// The most recent user message text (the salient §11 recall query); empty if none.
    fn latest_user_text(&self) -> String {
        self.snapshot
            .conversation
            .turns
            .iter()
            .rev()
            .find_map(|t| match t {
                Turn::User(u) => Some(u.text.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// §11 `before_compact` notification to every memory provider (a chance to persist before the
    /// context engine drops/summarizes turns).
    async fn before_compact_memory(&mut self) {
        for provider in &self.memory {
            provider.before_compact(&self.snapshot.conversation).await;
        }
    }

    /// §11 `after_turn` notification: hand every memory provider the just-recorded turn so it can
    /// persist new memories. Called after a turn's content is recorded, before `ctx.after_response`.
    async fn after_turn_memory(&self) {
        if self.memory.is_empty() {
            return;
        }
        if let Some(turn) = self.snapshot.conversation.turns.last().cloned() {
            for provider in &self.memory {
                provider
                    .after_turn(&turn, &self.snapshot.conversation)
                    .await;
            }
        }
    }

    /// §11 `on_session_switch` fan-out to every memory provider (a session-boundary consolidation
    /// chance). The reason distinguishes start/resume/compaction/handoff/end.
    async fn notify_session_switch(&self, reason: SwitchReason) {
        for provider in &self.memory {
            provider.on_session_switch(reason).await;
        }
    }

    /// The effective edit-approval policy (§12) for this session: the explicit per-session override
    /// on the snapshot, else the engine [`Config`] default.
    fn effective_policy(&self) -> crate::approval::ApprovalPolicy {
        self.snapshot
            .approval_policy
            .unwrap_or(self.config.approval_policy)
    }

    /// A best-effort description of the active model for the §10 [`on_model`](ContextEngine::on_model)
    /// hook: the profile label plus the provider's declared context window (if any).
    fn model_info(&self) -> crate::context::ModelInfo {
        crate::context::ModelInfo {
            model: self.profile.as_str().to_string(),
            max_context: self.provider.capabilities().max_context,
        }
    }

    /// Fire the once-per-incarnation §10/§11 lifecycle hooks before the first turn does any work:
    /// `context.on_model` -> `context.on_session_start` -> `memory.on_session_switch(Start|Resume)`.
    /// Idempotent — a no-op on every turn after the first. `resuming` reflects whether this
    /// incarnation re-activated on a background completion (so the boundary is a `Resume`, not a
    /// fresh `Start`).
    async fn ensure_session_started(&mut self, resuming: bool) {
        if self.lifecycle_started {
            return;
        }
        self.lifecycle_started = true;
        let info = self.model_info();
        self.context.on_model(&info);
        self.context.on_session_start(&self.snapshot.session_id);
        let reason = if resuming {
            SwitchReason::Resume
        } else {
            SwitchReason::Start
        };
        self.notify_session_switch(reason).await;
    }

    /// End the session: notify the §10 context engine and §11 memory providers so they can flush /
    /// consolidate. A host calls this on incarnation teardown (terminal deactivation). The context
    /// engine sees the final conversation; memory providers get `on_session_switch(End)`.
    pub async fn end_session(&mut self) {
        self.context
            .on_session_end(&self.snapshot.session_id, &self.snapshot.conversation);
        self.notify_session_switch(SwitchReason::End).await;
    }

    /// §10/§11 pre-turn hooks (run once before the ReAct loop): re-gather memory recall/blocks into
    /// the §10 [`PromptAssembler`] tiers, then measure budget [`Pressure`](crate::context::Pressure)
    /// and proactively compact when over the configured budget (`memory.before_compact` ->
    /// `ctx.compact`). Memory population is a no-op until a [`MemoryProvider`](crate::memory::MemoryProvider)
    /// is registered.
    async fn prepare_turn_context(&mut self, events: &EventSink) {
        self.assembler.reset_turn();
        self.gather_memory().await;
        // §10 generic stable-tier blocks (e.g. the skills index), folded after memory blocks. Each
        // source is expected to be cache-stable so the system prompt stays byte-stable across turns.
        for source in &self.prompt_sources {
            if let Some(block) = source.block() {
                if !block.is_empty() {
                    self.assembler.stable.push(block);
                }
            }
        }
        let budget = self.config.context_budget_tokens.map(|b| b as usize);
        let pressure = self
            .context
            .before_turn(&self.snapshot.conversation, budget);
        let max_context = self.provider.capabilities().max_context.map(|c| c as u64);
        // Compact to the context engine's *effective* budget: the host `context_budget_tokens`
        // override when set, else the engine's own threshold (LCM sizes one from the model window in
        // `on_model`). The budgeted default returns `budget_tokens == budget`, so this is a no-op
        // change for it (None => never over budget).
        let mut compacted = false;
        let mut dropped_turns = 0u32;
        if let (true, Some(target)) = (pressure.over_budget(), pressure.budget_tokens) {
            self.before_compact_memory().await;
            let before = self.snapshot.conversation.turns.len();
            let conv = std::mem::take(&mut self.snapshot.conversation);
            self.snapshot.conversation = self.context.compact(conv, target).await;
            dropped_turns = before.saturating_sub(self.snapshot.conversation.turns.len()) as u32;
            compacted = dropped_turns > 0;
            self.notify_session_switch(SwitchReason::Compaction).await;
        }
        // Re-measure after a possible compaction so the HUD reflects the context the turn will use.
        let used = crate::context::estimate_tokens(&self.snapshot.conversation) as u64;
        events.emit(|seq| AgentEvent::Context {
            seq,
            status: ContextStatus {
                used_tokens: used,
                max_tokens: max_context,
                budget_tokens: pressure.budget_tokens.map(|b| b as u64),
                compacted,
                dropped_turns,
            },
        });
    }

    /// Compact the conversation via the §10 context engine (the §8 -> §10 tie-in). On an explicit
    /// `ContextOverflow`/`PayloadTooLarge` we compact regardless of the soft budget: use the
    /// configured budget if set, else force a reduction by targeting half the current estimate.
    /// Returns whether any turn was actually dropped (the recovery loop treats `false` as
    /// "overflow is unrecoverable").
    async fn compact_context(&mut self, events: &EventSink) -> bool {
        let target = self
            .config
            .context_budget_tokens
            .map(|b| b as usize)
            .unwrap_or_else(|| crate::context::estimate_tokens(&self.snapshot.conversation) / 2);
        let before = self.snapshot.conversation.turns.len();
        let conv = std::mem::take(&mut self.snapshot.conversation);
        self.snapshot.conversation = self.context.compact(conv, target).await;
        let after = self.snapshot.conversation.turns.len();
        let dropped = after < before;
        if dropped {
            self.notify_session_switch(SwitchReason::Compaction).await;
            let used = crate::context::estimate_tokens(&self.snapshot.conversation) as u64;
            events.emit(|seq| AgentEvent::Context {
                seq,
                status: ContextStatus {
                    used_tokens: used,
                    max_tokens: self.provider.capabilities().max_context.map(|c| c as u64),
                    budget_tokens: self.config.context_budget_tokens.map(|b| b as u64),
                    compacted: true,
                    dropped_turns: (before - after) as u32,
                },
            });
        }
        dropped
    }

    /// Finalize a text-only turn: append the assistant turn. The text/reasoning deltas were already
    /// streamed to the host by [`Engine::call_model`] (via [`drive_model_call`]); this only records
    /// the durable conversation turn.
    fn finalize_text(&mut self, out: &ModelOutput, _events: &EventSink) {
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
            // Resume path: a background completion arrived. First resolve any parked **approval**
            // decisions (§12 HITL) — re-running the approved tool call (allow) or injecting a
            // tool-error (deny) — taking those completions out of the pending set; then apply the
            // remaining delegation completions and fall into the ReAct loop so the model sees the
            // resolved tool result(s).
            if !self.snapshot.pending_approvals.is_empty() {
                self.resolve_approvals(host, events, control).await;
            }
            self.resolve_pending();
            if let Some(next) = self.snapshot.pending_approvals.first().map(|a| a.job_id.clone()) {
                // Not every parked approval was answered yet: re-suspend deterministically on a
                // remaining one (the operator answers them one at a time).
                self.snapshot.waiting_for = self
                    .snapshot
                    .pending_approvals
                    .iter()
                    .map(|a| a.job_id.clone())
                    .collect();
                return Ok(self.suspend_for_approval(next, events, false));
            }
            self.snapshot.waiting_for.clear();
            self.snapshot.epoch = self.snapshot.epoch.next();
        } else if let Some(job_id) = self.snapshot.waiting_for.first().cloned() {
            // Re-activated while still suspended (e.g. recovery before the worker ran): re-suspend the
            // same job deterministically; the durable outbox dedupes the re-enqueue. An unanswered
            // approval park re-parks (no enqueue), everything else re-enqueues the background job.
            if !self.snapshot.pending_approvals.is_empty() {
                return Ok(self.suspend_for_approval(job_id, events, false));
            }
            return Ok(self.suspend(job_id, events, false));
        }

        // §10/§11 once-per-incarnation lifecycle hooks before the first turn's work.
        self.ensure_session_started(resuming).await;

        // §11/§10 pre-turn hooks: gather memory recall/blocks into the assembler, then measure
        // budget pressure and compact if over budget (memory.before_compact -> ctx.compact).
        self.prepare_turn_context(events).await;

        // The in-turn ReAct loop (§4.2): build_context -> call_model -> execute_tools -> observe ->
        // call_model again, until the model returns final text (completion) or a tool delegates
        // (durable suspension). The iteration budget is the hard stop. The loop runs fully
        // in-process within one durable turn; only `Effect::Delegate` crosses the suspension boundary.
        let exec = self.exec.clone();
        let registry = self.registry.clone();
        let tool_result_budget = self.config.tool_result_budget;
        let effective_policy = self.effective_policy();
        let mut rounds_left = self.config.max_iterations;
        // Accumulated usage across every model call this turn makes (each round + the final summary
        // call), so `TurnSummary.usage` is the turn total, not just the last call's delta.
        let mut turn_usage = UsageDelta::default();
        // Engine-native post-turn review nudge bookkeeping (§4.3): tool-executing rounds this turn,
        // and whether a skill/memory tool ran (resets the corresponding cadence counter, mirroring
        // hermes `tool_executor.py` resetting `_iters_since_skill` on `skill_manage`).
        let mut tool_rounds: u32 = 0;
        let mut used_skill_tool = false;
        let mut used_memory_tool = false;

        let cancel = control.cancel_token();
        loop {
            if rounds_left == 0 {
                // Budget exhausted: one final toolless call asks the model to summarize what it has,
                // then the turn ends `BudgetExhausted` (the model cannot keep calling tools forever).
                return Ok(self
                    .finish_budget_exhausted(events, &cancel, turn_usage)
                    .await);
            }
            rounds_left -= 1;

            let out = match self.call_model(events, true, &cancel).await {
                Ok(out) => out,
                Err(f) => return Ok(self.finish_failed(f, events)),
            };
            turn_usage.add(&out.usage);

            // Boundary after the model call: serve snapshots/steer, honor a mid-call interrupt.
            if self.boundary(control, events) {
                return Ok(self.finish_interrupted(events));
            }

            if out.tool_calls.is_empty() {
                // §11 -> §10 post-turn hooks (spec order): record the assistant turn, then
                // memory.after_turn, then ctx.after_response.
                self.finalize_text(&out, events);
                self.after_turn_memory().await;
                self.context.after_response(&out.usage);
                // §4.3 engine-native post-turn trigger: advance the review cadence counters and, on
                // a threshold, fire-and-forget a background-review child (no suspend).
                self.maybe_emit_reviews(host, tool_rounds, used_skill_tool, used_memory_tool)
                    .await;
                return Ok(self.complete(out, events, turn_usage));
            }

            let cx = TurnCx {
                cancel: control.cancel_token(),
                events,
                host,
                session_id: self.snapshot.session_id.clone(),
                budget: self.budget,
                exec: &*exec,
                tool_result_budget,
                approval_policy: effective_policy,
                pre_approved: false,
            };

            // execute_tools: run the model's tool batch through the §12 pipeline, collecting result
            // slots, effects, and structured detail for the rich §17 transcript views.
            //
            // A batch runs **concurrently** only when it has more than one call and *every* call
            // resolves to a tool that declares itself [`ToolConcurrency::Parallel`] (all-or-nothing;
            // any exclusive/mutating call serializes the batch — see [`crate::tools::Tool::concurrency`]).
            let view_of = |call: &crate::conversation::ToolCall| ToolCallView {
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
            let result_view_of = |outcome: &crate::tools::ToolOutcome| ToolResultView {
                call_id: outcome.result.call_id.clone(),
                ok: outcome.result.ok,
                summary: outcome.result.content.clone(),
                // The tool's typed output (fs listing, command exit/stdout, ...) for a rich
                // consumer; `None` for plain-text tools.
                detail: outcome.detail.clone(),
            };

            let mut calls = Vec::new();
            let mut effects: Vec<Effect> = Vec::new();
            let mut interrupted = false;

            // Count this tool-executing round and note skill/memory tool use for the review nudges.
            tool_rounds = tool_rounds.saturating_add(1);
            for c in &out.tool_calls {
                if c.name.starts_with("skill_manage") {
                    used_skill_tool = true;
                } else if c.name.starts_with("mnemosyne_") {
                    used_memory_tool = true;
                }
            }

            let parallel = out.tool_calls.len() > 1
                && out.tool_calls.iter().all(|c| {
                    registry
                        .get(&c.name)
                        .map(|t| t.concurrency() == crate::tools::ToolConcurrency::Parallel)
                        .unwrap_or(false)
                });

            if parallel {
                // Emit all starts in call order, run the batch concurrently, then drain results in
                // call order. Read-only parallel tools have no ordered side effects, so the boundary
                // is evaluated once after the whole batch settles.
                for call in &out.tool_calls {
                    let view = view_of(call);
                    events.emit(|seq| AgentEvent::ToolStarted { seq, call: view });
                }
                let outcomes = futures::future::join_all(
                    out.tool_calls
                        .iter()
                        .map(|call| async { (call.clone(), run_tool(call, &registry, &cx).await) }),
                )
                .await;
                for (call, outcome) in outcomes {
                    let result_view = result_view_of(&outcome);
                    events.emit(|seq| AgentEvent::ToolFinished {
                        seq,
                        result: result_view,
                    });
                    calls.push((call, outcome.result));
                    effects.extend(outcome.effects);
                }
                if self.boundary_readonly(control, events) {
                    interrupted = true;
                }
            } else {
                for call in &out.tool_calls {
                    let view = view_of(call);
                    events.emit(|seq| AgentEvent::ToolStarted { seq, call: view });
                    let outcome = run_tool(call, &registry, &cx).await;
                    let result_view = result_view_of(&outcome);
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
            let mut spawns: Vec<daemon_protocol::SpawnSpec> = Vec::new();
            let mut awaiting: Vec<crate::snapshot::PendingApproval> = Vec::new();
            for effect in effects {
                match effect {
                    Effect::Persist(turn) => self.snapshot.conversation.turns.push(turn),
                    Effect::Delegate(job_id) => delegated = Some(job_id),
                    Effect::Spawn(spec) => spawns.push(spec),
                    Effect::AwaitDecision {
                        job_id,
                        call,
                        prompt,
                        path,
                    } => awaiting.push(crate::snapshot::PendingApproval {
                        job_id,
                        call,
                        prompt,
                        path,
                    }),
                }
            }
            // Fire-and-forget: issue each spawn as a non-joining host request and keep running. The
            // parent never enters `waiting_for` and never suspends for these (cf. `Delegate` below).
            for spec in spawns {
                self.issue_spawn(host, spec).await;
            }

            // An interrupt at a tool boundary finalizes the turn before it would suspend/loop.
            if interrupted {
                return Ok(self.finish_interrupted(events));
            }

            if !awaiting.is_empty() {
                // A gated tool needs a durable operator decision (§12 HITL): record the parked
                // approvals on the snapshot and suspend, waiting on the operator's answer (delivered
                // as a wake completion). Mirrors delegation suspension, but the wake source is an
                // operator (`ApprovalDecide`), not a background worker.
                self.notify_session_switch(SwitchReason::Handoff).await;
                let first = awaiting[0].job_id.clone();
                for a in &awaiting {
                    self.snapshot.waiting_for.push(a.job_id.clone());
                }
                self.snapshot.pending_approvals.extend(awaiting);
                return Ok(self.suspend_for_approval(first, events, true));
            }

            if let Some(job_id) = delegated {
                // A delegation crosses the durable boundary: notify memory of the handoff, then
                // suspend the turn and wait for the wake.
                self.notify_session_switch(SwitchReason::Handoff).await;
                self.snapshot.waiting_for.push(job_id.clone());
                return Ok(self.suspend(job_id, events, true));
            }
            // §11 -> §10 post-round hooks (spec order) on the recorded tool turn, then loop — the
            // next `call_model` sees the tool results in context.
            self.after_turn_memory().await;
            self.context.after_response(&out.usage);
        }
    }

    /// Finalize a turn that hit its iteration budget: one final toolless model call to summarize,
    /// then `TurnFinished { BudgetExhausted }`. Any tool calls the model attempts on this pass are
    /// ignored — the turn is ending.
    async fn finish_budget_exhausted(
        &mut self,
        events: &EventSink,
        cancel: &CancellationToken,
        mut turn_usage: UsageDelta,
    ) -> TurnOutcome {
        let out = match self.call_model(events, false, cancel).await {
            Ok(out) => out,
            Err(f) => return self.finish_failed(f, events),
        };
        self.finalize_text(&out, events);
        turn_usage.add(&out.usage);
        let summary = TurnSummary {
            end_reason: EndReason::BudgetExhausted,
            final_text: Some(out.text),
            usage: turn_usage,
        };
        let emitted = summary.clone();
        events.emit(|seq| AgentEvent::TurnFinished {
            seq,
            summary: emitted,
        });
        TurnOutcome::Completed(summary)
    }

    /// Emit the terminal `TurnFinished` and build the completed outcome. `turn_usage` is the folded
    /// usage of every model call this turn made (the per-call deltas were already streamed live).
    fn complete(&self, out: ModelOutput, events: &EventSink, turn_usage: UsageDelta) -> TurnOutcome {
        let summary = TurnSummary {
            end_reason: EndReason::Completed,
            final_text: Some(out.text),
            usage: turn_usage,
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

    /// Suspend the turn awaiting a durable operator approval decision (§12 HITL). Like [`suspend`]
    /// it emits the suspending `TurnFinished` (bumping the epoch on a fresh park, not on a
    /// deterministic recovery re-park), but the payload marks it an approval park (`await-approval`)
    /// so the host parks it for an operator — never enqueuing a runnable background job.
    fn suspend_for_approval(
        &mut self,
        job_id: JobId,
        events: &EventSink,
        bump_epoch: bool,
    ) -> TurnOutcome {
        if bump_epoch {
            self.snapshot.epoch = self.snapshot.epoch.next();
        }
        let summary = TurnSummary::ended(EndReason::Suspended);
        events.emit(|seq| AgentEvent::TurnFinished { seq, summary });
        TurnOutcome::Suspended(Suspension {
            job_id,
            epoch: self.snapshot.epoch,
            payload: APPROVAL_SUSPEND_PAYLOAD.to_vec(),
        })
    }

    /// Resolve parked §12 approval decisions on resume: for each unapplied completion whose `job_id`
    /// matches a [`PendingApproval`](crate::snapshot::PendingApproval), re-run the approved tool call
    /// (allow — performs the side effect, with `pre_approved` set so the tool skips its gate) or
    /// inject a tool-error (deny), then splice the result into the parked `awaiting-approval` slot.
    /// Approval completions are taken out of `self.pending` so the delegation resolver ignores them.
    async fn resolve_approvals(
        &mut self,
        host: &dyn HostRequestHandler,
        events: &EventSink,
        control: &TurnControl,
    ) {
        let exec = self.exec.clone();
        let registry = self.registry.clone();
        let budget = self.budget;
        let tool_result_budget = self.config.tool_result_budget;
        let policy = self.effective_policy();
        let session_id = self.snapshot.session_id.clone();
        let cancel = control.cancel_token();
        let pending = std::mem::take(&mut self.pending);
        let mut rest = Vec::new();
        for completion in pending {
            match self
                .snapshot
                .pending_approvals
                .iter()
                .position(|p| p.job_id == completion.job_id)
            {
                Some(i) => {
                    let approval = self.snapshot.pending_approvals.remove(i);
                    let decision = String::from_utf8_lossy(&completion.payload);
                    let allow = decision.starts_with("allow");
                    let (ok, content) = if allow {
                        let cx = TurnCx {
                            cancel: cancel.clone(),
                            events,
                            host,
                            session_id: session_id.clone(),
                            budget,
                            exec: &*exec,
                            tool_result_budget,
                            approval_policy: policy,
                            pre_approved: true,
                        };
                        let outcome = run_tool(&approval.call, &registry, &cx).await;
                        (outcome.result.ok, outcome.result.content)
                    } else {
                        (
                            false,
                            format!("operator denied this action (request {})", approval.job_id),
                        )
                    };
                    self.replace_awaiting_result(&approval.job_id, ok, content);
                }
                None => rest.push(completion),
            }
        }
        self.pending = rest;
    }

    /// Splice a resolved approval result into the conversation slot the parked tool call left behind
    /// (its content is the `awaiting-approval:{job_id}` marker).
    fn replace_awaiting_result(&mut self, job_id: &JobId, ok: bool, content: String) {
        let marker = format!("awaiting-approval:{job_id}");
        for turn in self.snapshot.conversation.turns.iter_mut() {
            if let Turn::Tool(tool_turn) = turn {
                for (_call, result) in tool_turn.calls.iter_mut() {
                    if result.content.contains(&marker) {
                        result.ok = ok;
                        result.content = content;
                        return;
                    }
                }
            }
        }
    }

    /// Advance the post-turn review cadence counters and emit background-review spawns on threshold
    /// (§4.3). Skill review is paced in tool iterations (reset on `skill_manage` use); memory review
    /// in completed turns (reset on a memory write). A `0` interval disables that review. Counters
    /// live in the durable [`Snapshot`] so the cadence survives suspension. Mirrors hermes'
    /// `turn_finalizer.py:375-401` nudge gates without forking on the parent's thread.
    async fn maybe_emit_reviews(
        &mut self,
        host: &dyn HostRequestHandler,
        tool_rounds: u32,
        used_skill_tool: bool,
        used_memory_tool: bool,
    ) {
        let mut spawns: Vec<SpawnSpec> = Vec::new();

        // Skill review: count this turn's tool iterations, but a `skill_manage` use this turn resets
        // the cadence (the agent just curated skills — no nudge needed).
        self.snapshot.iters_since_skill = self
            .snapshot
            .iters_since_skill
            .saturating_add(tool_rounds);
        if used_skill_tool {
            self.snapshot.iters_since_skill = 0;
        } else if self.config.skill_review_interval > 0
            && self.snapshot.iters_since_skill >= self.config.skill_review_interval
        {
            self.snapshot.iters_since_skill = 0;
            spawns.push(SpawnSpec {
                kind: "skill_review".to_owned(),
                seed: SpawnSeed::FromConversation,
            });
        }

        // Memory review: one completed turn, reset by a memory write this turn.
        self.snapshot.turns_since_memory = self.snapshot.turns_since_memory.saturating_add(1);
        if used_memory_tool {
            self.snapshot.turns_since_memory = 0;
        } else if self.config.memory_review_interval > 0
            && self.snapshot.turns_since_memory >= self.config.memory_review_interval
        {
            self.snapshot.turns_since_memory = 0;
            spawns.push(SpawnSpec {
                kind: "memory_review".to_owned(),
                seed: SpawnSeed::FromConversation,
            });
        }

        for spec in spawns {
            self.issue_spawn(host, spec).await;
        }
    }

    /// Issue a fire-and-forget [`HostRequestKind::Spawn`] for an attached, non-joining background
    /// child (§4.3). Unlike a delegation this does **not** touch `waiting_for` or suspend: the host
    /// records the parent->child edge for audit and runs the child to its own terminal state. The
    /// returned child id is purely informational; an unknown `kind` is a host-side no-op.
    async fn issue_spawn(&self, host: &dyn HostRequestHandler, spec: SpawnSpec) {
        let _ = host
            .request(HostRequest {
                request_id: ReqId(0),
                kind: HostRequestKind::Spawn { spec },
            })
            .await;
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

    /// A provider that always completes with plain final text (no tool calls).
    struct TextProvider;

    #[async_trait::async_trait]
    impl Provider for TextProvider {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }
        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            Ok(ModelOutput {
                text: "ok".into(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage: UsageDelta::default(),
            })
        }
    }

    /// A provider that records the system prompt of the request it receives, then completes.
    struct SystemRecordingProvider {
        seen: std::sync::Mutex<String>,
    }

    #[async_trait::async_trait]
    impl Provider for SystemRecordingProvider {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }
        async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
            *self.seen.lock().unwrap() = req.system.clone();
            Ok(ModelOutput {
                text: "ok".into(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage: UsageDelta::default(),
            })
        }
    }

    /// A stable-tier source emitting a fixed block.
    struct FixedBlock(&'static str);
    impl crate::context::StablePromptSource for FixedBlock {
        fn block(&self) -> Option<String> {
            Some(self.0.to_string())
        }
    }

    /// A registered [`StablePromptSource`] is folded into the request's system preamble each turn.
    #[tokio::test]
    async fn prompt_source_block_is_injected_into_system() {
        let provider = Arc::new(SystemRecordingProvider {
            seen: std::sync::Mutex::new(String::new()),
        });
        let mut engine = Engine::fresh(
            SessionId::new("ps"),
            SystemPrompt::new("base system"),
            provider.clone(),
            Arc::new(ToolRegistry::new()),
        )
        .with_prompt_sources(vec![Arc::new(FixedBlock("<available_skills>\n  x\n</available_skills>"))]);
        engine.push_user(UserMsg::new("hi"));
        engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .unwrap();
        let seen = provider.seen.lock().unwrap().clone();
        assert!(seen.contains("base system"), "keeps the base system prompt");
        assert!(seen.contains("<available_skills>"), "folds the stable block in");
    }

    /// A host that records every spawn `kind` it is asked to materialize.
    #[derive(Default)]
    struct SpawnRecordingHost {
        spawns: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl HostRequestHandler for SpawnRecordingHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            if let HostRequestKind::Spawn { spec } = &req.kind {
                self.spawns.lock().unwrap().push(spec.kind.clone());
                return HostResponse {
                    request_id: req.request_id,
                    body: HostResponseBody::Spawned(SessionId::new("child")),
                };
            }
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved(true),
            }
        }
    }

    /// With `memory_review_interval = 1`, a completed turn (no memory write) fires exactly one
    /// fire-and-forget `memory_review` spawn and the turn still completes normally (no suspend).
    #[tokio::test]
    async fn memory_review_nudge_emits_spawn_on_threshold() {
        let host = SpawnRecordingHost::default();
        let mut engine = Engine::fresh(
            SessionId::new("nudge"),
            SystemPrompt::new("test"),
            Arc::new(TextProvider),
            Arc::new(ToolRegistry::new()),
        )
        .with_config(Config {
            memory_review_interval: 1,
            ..Config::default()
        });
        engine.push_user(UserMsg::new("hi"));

        let outcome = engine
            .run_turn(&host, &EventSink::discarding(), &TurnControl::new())
            .await
            .expect("turn completes");
        assert!(matches!(outcome, TurnOutcome::Completed(_)));
        assert!(engine.snapshot().waiting_for.is_empty(), "spawn does not suspend");
        assert_eq!(
            *host.spawns.lock().unwrap(),
            vec!["memory_review".to_string()],
        );
        assert_eq!(engine.snapshot().turns_since_memory, 0, "counter reset");
    }

    /// The default `0` intervals disable the engine-native trigger entirely.
    #[tokio::test]
    async fn review_nudges_disabled_by_default() {
        let host = SpawnRecordingHost::default();
        let mut engine = Engine::fresh(
            SessionId::new("no-nudge"),
            SystemPrompt::new("test"),
            Arc::new(TextProvider),
            Arc::new(ToolRegistry::new()),
        );
        engine.push_user(UserMsg::new("hi"));
        engine
            .run_turn(&host, &EventSink::discarding(), &TurnControl::new())
            .await
            .expect("turn completes");
        assert!(host.spawns.lock().unwrap().is_empty(), "no spawns when disabled");
    }

    /// A credential provider serving two profiles with distinct secrets, recording each acquisition
    /// — so a test can observe the engine hopping from the primary to the fallback profile.
    struct TwoProfileCreds {
        acquired: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl crate::credentials::CredentialProvider for TwoProfileCreds {
        async fn acquire(
            &self,
            profile: &ProfileRef,
            scope: &CredScope,
        ) -> Result<daemon_common::CapabilityLease, daemon_common::CredError> {
            self.acquired.lock().unwrap().push(profile.to_string());
            Ok(daemon_common::CapabilityLease {
                cap_id: daemon_common::CredId::new(format!("{profile}-cap")),
                profile: profile.clone(),
                scope: scope.clone(),
                mode: daemon_common::CredMode::Native,
                expires_at_ms: crate::credentials::now_ms() + 60_000,
                secret: Some(daemon_common::LeaseSecret::new(format!("sk-{profile}"))),
                signature: Vec::new(),
            })
        }
        async fn release(&self, _lease: &daemon_common::CapabilityLease) {}
        async fn rotate(&self, _profile: &ProfileRef, _cap_id: &daemon_common::CredId) {}
    }

    /// A provider that rejects every credential with a content-policy failure *except* the fallback
    /// profile's secret — so the turn only completes once the engine has hopped profiles.
    struct PolicyGatedProvider {
        ok_secret: String,
    }

    #[async_trait::async_trait]
    impl Provider for PolicyGatedProvider {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }

        async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
            if req.auth.as_deref() == Some(self.ok_secret.as_str()) {
                Ok(ModelOutput {
                    text: "ok on fallback profile".into(),
                    reasoning: None,
                    tool_calls: Vec::new(),
                    usage: UsageDelta::default(),
                })
            } else {
                Err(Failure::ContentPolicy("blocked on primary".into()))
            }
        }
    }

    /// A persistent content-policy failure hops once to the configured fallback profile (wired via
    /// `EngineProfile::with_fallback_profile`); the engine then completes on the fallback's
    /// credential, having acquired the primary first and the fallback second.
    #[tokio::test]
    async fn fallback_profile_hops_credential_profile() {
        use crate::EngineProfile;
        use std::sync::Arc;

        let creds = Arc::new(TwoProfileCreds {
            acquired: std::sync::Mutex::new(Vec::new()),
        });
        let creds_for_builder = creds.clone();
        let profile = EngineProfile::new(
            Arc::new(|| {
                Arc::new(PolicyGatedProvider {
                    ok_secret: "sk-fallback".into(),
                }) as Arc<dyn Provider>
            }),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("test"),
        )
        .with_credentials(
            Arc::new(move || creds_for_builder.clone() as Arc<dyn crate::credentials::CredentialProvider>),
            ProfileRef::new("primary"),
        )
        .with_fallback_profile(ProfileRef::new("fallback"));

        let mut engine = profile.fresh(SessionId::new("fallback-hop"));
        engine.push_user(UserMsg::new("hello"));

        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .expect("turn completes after hopping to the fallback profile");
        assert!(matches!(outcome, TurnOutcome::Completed(_)));
        let acquired = creds.acquired.lock().unwrap().clone();
        assert_eq!(
            acquired,
            vec!["primary".to_string(), "fallback".to_string()],
            "acquired the primary first, then hopped to the fallback profile"
        );
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

    fn looping_engine(
        provider: Arc<dyn Provider>,
        runs: Arc<AtomicU64>,
        max_iterations: u32,
    ) -> Engine {
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
        assert_eq!(
            runs.load(Ordering::Relaxed),
            2,
            "the tool ran on both rounds"
        );
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

    /// A tool that records the *peak* number of concurrent in-flight executions across a batch. By
    /// holding a short overlap window in `run`, two genuinely concurrent calls observe `max_seen == 2`
    /// while serialized calls only ever observe `1` — a deterministic probe for §12 batch concurrency.
    struct ProbeTool {
        name: &'static str,
        concurrency: crate::tools::ToolConcurrency,
        active: Arc<AtomicU64>,
        max_seen: Arc<AtomicU64>,
    }

    #[async_trait::async_trait]
    impl Tool for ProbeTool {
        fn name(&self) -> &str {
            self.name
        }
        fn schema(&self) -> &str {
            "{}"
        }
        fn concurrency(&self) -> crate::tools::ToolConcurrency {
            self.concurrency
        }
        async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> crate::tools::ToolOutcome {
            let cur = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_seen.fetch_max(cur, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            crate::tools::ToolOutcome::text(call.call_id.clone(), true, "ok")
        }
    }

    fn probe_engine(provider: Arc<dyn Provider>, tools: Vec<Arc<dyn Tool>>) -> Engine {
        let mut registry = ToolRegistry::new();
        for t in tools {
            registry.register(t);
        }
        Engine::fresh(
            SessionId::new("probe"),
            SystemPrompt::new("test"),
            provider,
            Arc::new(registry),
        )
        .with_config(Config {
            max_iterations: 8,
            ..Config::default()
        })
    }

    /// A batch of two `Parallel` tool calls runs concurrently: the peak observed in-flight count is 2.
    #[tokio::test]
    async fn parallel_tool_batch_runs_concurrently() {
        let active = Arc::new(AtomicU64::new(0));
        let max_seen = Arc::new(AtomicU64::new(0));
        let provider = Arc::new(ScriptedProvider::new(
            vec![ScriptStep::Calls(vec![
                ("para".into(), "{}".into()),
                ("para".into(), "{}".into()),
            ])],
            "done",
        ));
        let tool = Arc::new(ProbeTool {
            name: "para",
            concurrency: crate::tools::ToolConcurrency::Parallel,
            active: active.clone(),
            max_seen: max_seen.clone(),
        });
        let mut engine = probe_engine(provider, vec![tool]);
        engine.push_user(UserMsg::new("go"));

        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .unwrap();
        assert!(matches!(outcome, TurnOutcome::Completed(_)));
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            2,
            "both parallel calls should be in flight at once"
        );
    }

    /// A mixed batch (one `Parallel`, one `Exclusive`) is serialized all-or-nothing: peak in-flight is 1.
    #[tokio::test]
    async fn exclusive_call_serializes_the_batch() {
        let active = Arc::new(AtomicU64::new(0));
        let max_seen = Arc::new(AtomicU64::new(0));
        let provider = Arc::new(ScriptedProvider::new(
            vec![ScriptStep::Calls(vec![
                ("para".into(), "{}".into()),
                ("excl".into(), "{}".into()),
            ])],
            "done",
        ));
        let para = Arc::new(ProbeTool {
            name: "para",
            concurrency: crate::tools::ToolConcurrency::Parallel,
            active: active.clone(),
            max_seen: max_seen.clone(),
        });
        let excl = Arc::new(ProbeTool {
            name: "excl",
            concurrency: crate::tools::ToolConcurrency::Exclusive,
            active: active.clone(),
            max_seen: max_seen.clone(),
        });
        let mut engine = probe_engine(provider, vec![para, excl]);
        engine.push_user(UserMsg::new("go"));

        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .unwrap();
        assert!(matches!(outcome, TurnOutcome::Completed(_)));
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            1,
            "an exclusive call must force the whole batch to run sequentially"
        );
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

    // --- §8 recovery + §10/§11 hooks ---

    use std::collections::VecDeque;

    /// A provider that replays a scripted sequence of results (failures then success), defaulting to
    /// a completing response once the script is exhausted.
    struct FaultProvider {
        script: std::sync::Mutex<VecDeque<Result<ModelOutput, Failure>>>,
        calls: AtomicU64,
    }

    impl FaultProvider {
        fn new(seq: Vec<Result<ModelOutput, Failure>>) -> Self {
            Self {
                script: std::sync::Mutex::new(seq.into_iter().collect()),
                calls: AtomicU64::new(0),
            }
        }
    }

    fn ok_text(text: &str) -> ModelOutput {
        ModelOutput {
            text: text.into(),
            reasoning: None,
            tool_calls: Vec::new(),
            usage: UsageDelta::default(),
        }
    }

    #[async_trait::async_trait]
    impl Provider for FaultProvider {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }
        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.script
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Ok(ok_text("default")))
        }
    }

    /// A 429 with backoff retries (emitting `RateLimit`) and then completes on a fresh attempt.
    #[tokio::test]
    async fn rate_limit_retries_with_backoff_then_completes() {
        let provider = Arc::new(FaultProvider::new(vec![
            Err(Failure::RateLimit {
                retry_after: None,
                message: "slow down".into(),
            }),
            Err(Failure::RateLimit {
                retry_after: None,
                message: "slow down".into(),
            }),
            Ok(ok_text("recovered")),
        ]));
        // Tiny backoff so the test is fast.
        let config = Config {
            model_backoff_base_ms: 1,
            model_backoff_max_ms: 2,
            model_max_retries: 3,
            ..Config::default()
        };
        let mut engine = Engine::fresh(
            SessionId::new("rl"),
            SystemPrompt::new("test"),
            provider.clone(),
            Arc::new(ToolRegistry::new()),
        )
        .with_config(config);
        engine.push_user(UserMsg::new("hello"));
        let (sink, log) = collecting();

        let outcome = engine
            .run_turn(&NoopHost, &sink, &TurnControl::new())
            .await
            .unwrap();
        match outcome {
            TurnOutcome::Completed(s) => {
                assert_eq!(s.end_reason, EndReason::Completed);
                assert_eq!(s.final_text.as_deref(), Some("recovered"));
            }
            _ => panic!("expected completion after backoff retries"),
        }
        assert_eq!(
            provider.calls.load(Ordering::Relaxed),
            3,
            "two retries then success"
        );
        assert!(
            log.lock()
                .unwrap()
                .iter()
                .any(|e| matches!(e, AgentEvent::RateLimit { .. })),
            "a RateLimit event was emitted during backoff"
        );
    }

    /// A `ContextOverflow` compacts the conversation once (the §8 -> §10 tie-in) then retries and
    /// completes; the conversation is shorter afterwards.
    #[tokio::test]
    async fn context_overflow_compacts_then_retries() {
        let provider = Arc::new(FaultProvider::new(vec![
            Err(Failure::ContextOverflow("too long".into())),
            Ok(ok_text("after compact")),
        ]));
        let mut engine = Engine::fresh(
            SessionId::new("overflow"),
            SystemPrompt::new("test"),
            provider.clone(),
            Arc::new(ToolRegistry::new()),
        );
        // Enough turns that drop-oldest frees > 10%.
        for i in 0..8 {
            engine.push_user(UserMsg::new(format!("message {i} ").repeat(20)));
        }
        let before = engine.snapshot().conversation.turns.len();

        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .unwrap();
        match outcome {
            TurnOutcome::Completed(s) => assert_eq!(s.final_text.as_deref(), Some("after compact")),
            _ => panic!("expected completion after compaction"),
        }
        assert_eq!(
            provider.calls.load(Ordering::Relaxed),
            2,
            "overflow then retry"
        );
        assert!(
            engine.snapshot().conversation.turns.len() < before,
            "the conversation was compacted"
        );
    }

    /// An always-overflowing provider compacts once then aborts (no infinite loop): the turn ends
    /// `Failed` rather than hanging.
    #[tokio::test]
    async fn unrecoverable_overflow_aborts() {
        let provider = Arc::new(FaultProvider::new(vec![
            Err(Failure::ContextOverflow("a".into())),
            Err(Failure::ContextOverflow("b".into())),
            Err(Failure::ContextOverflow("c".into())),
        ]));
        let mut engine = Engine::fresh(
            SessionId::new("overflow2"),
            SystemPrompt::new("test"),
            provider.clone(),
            Arc::new(ToolRegistry::new()),
        );
        for i in 0..8 {
            engine.push_user(UserMsg::new(format!("message {i} ").repeat(20)));
        }
        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .unwrap();
        match outcome {
            TurnOutcome::Completed(s) => assert_eq!(s.end_reason, EndReason::Failed),
            _ => panic!("expected a failed outcome, not a hang"),
        }
    }

    /// A context engine that always reports over-budget and drops one turn on compaction — used to
    /// prove the §10 compaction signal reaches the §17 stream as an `AgentEvent::Context`.
    struct DroppingContext;
    #[async_trait::async_trait]
    impl ContextEngine for DroppingContext {
        fn before_turn(
            &self,
            _conv: &Conversation,
            budget: Option<usize>,
        ) -> crate::context::Pressure {
            crate::context::Pressure {
                used_tokens: 1_000_000,
                budget_tokens: budget.or(Some(1)),
            }
        }
        async fn compact(&self, mut conv: Conversation, _budget: usize) -> Conversation {
            // Drop the oldest turn so the engine observes a non-zero `dropped_turns`.
            if !conv.turns.is_empty() {
                conv.turns.remove(0);
            }
            conv
        }
    }

    /// Compaction at the pre-turn pressure check emits an `AgentEvent::Context { compacted: true,
    /// dropped_turns >= 1 }` on the stream — the data a GUI renders as a "compacted" toast.
    #[tokio::test]
    async fn compaction_surfaces_context_event() {
        let mut engine = Engine::fresh(
            SessionId::new("compact-evt"),
            SystemPrompt::new("test"),
            Arc::new(crate::provider::MockProvider::completing("done")),
            Arc::new(ToolRegistry::new()),
        )
        .with_config(Config {
            context_budget_tokens: Some(1),
            ..Config::default()
        })
        .with_context_engine(Arc::new(DroppingContext));
        engine.push_user(UserMsg::new("first"));
        engine.push_user(UserMsg::new("second"));
        let (sink, log) = collecting();

        engine.run_turn(&NoopHost, &sink, &TurnControl::new()).await.unwrap();

        let log = log.lock().unwrap();
        assert!(
            log.iter().any(|e| matches!(
                e,
                AgentEvent::Context { status, .. } if status.compacted && status.dropped_turns >= 1
            )),
            "expected a compaction Context event, got: {log:?}"
        );
    }

    // §10/§11 hook-order instrumentation.
    struct RecordingContext {
        log: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }
    #[async_trait::async_trait]
    impl ContextEngine for RecordingContext {
        fn on_model(&self, _model: &crate::context::ModelInfo) {
            self.log.lock().unwrap().push("on_model");
        }
        fn before_turn(
            &self,
            _conv: &Conversation,
            budget: Option<usize>,
        ) -> crate::context::Pressure {
            self.log.lock().unwrap().push("before_turn");
            // Force over-budget so the compaction hooks fire.
            crate::context::Pressure {
                used_tokens: 10_000,
                budget_tokens: budget,
            }
        }
        async fn compact(&self, conv: Conversation, _budget: usize) -> Conversation {
            self.log.lock().unwrap().push("compact");
            conv
        }
        fn after_response(&self, _usage: &UsageDelta) {
            self.log.lock().unwrap().push("after_response");
        }
        fn on_session_start(&self, _session: &SessionId) {
            self.log.lock().unwrap().push("session_start");
        }
        fn on_session_end(&self, _session: &SessionId, _conv: &Conversation) {
            self.log.lock().unwrap().push("session_end");
        }
    }

    struct RecordingMemory {
        log: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }
    #[async_trait::async_trait]
    impl MemoryProvider for RecordingMemory {
        fn name(&self) -> &str {
            "rec"
        }
        fn prompt_block(&self) -> Option<crate::memory::PromptBlock> {
            self.log.lock().unwrap().push("prompt_block");
            None
        }
        async fn recall(&self, _q: &RecallQuery) -> Option<crate::memory::RecalledBlock> {
            self.log.lock().unwrap().push("recall");
            None
        }
        async fn after_turn(&self, _turn: &Turn, _conv: &Conversation) {
            self.log.lock().unwrap().push("after_turn");
        }
        async fn before_compact(&self, _conv: &Conversation) {
            self.log.lock().unwrap().push("before_compact");
        }
        async fn on_session_switch(&self, reason: SwitchReason) {
            let label = match reason {
                SwitchReason::Start => "switch:start",
                SwitchReason::Compaction => "switch:compaction",
                SwitchReason::Handoff => "switch:handoff",
                SwitchReason::Resume => "switch:resume",
                SwitchReason::End => "switch:end",
                SwitchReason::Manual => "switch:manual",
            };
            self.log.lock().unwrap().push(label);
        }
    }

    /// The §10/§11 hooks fire in spec order across an incarnation: the once-per-incarnation
    /// lifecycle hooks (`on_model -> session_start -> switch:start`) precede the per-turn hooks
    /// (`recall -> prompt_block -> before_turn -> before_compact -> compact -> switch:compaction ->
    /// after_turn -> after_response`), and `end_session` fires the teardown hooks
    /// (`session_end -> switch:end`).
    #[tokio::test]
    async fn memory_and_context_hooks_fire_in_spec_order() {
        let log = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
        let config = Config {
            context_budget_tokens: Some(1),
            ..Config::default()
        };
        let mut engine = Engine::fresh(
            SessionId::new("hooks"),
            SystemPrompt::new("test"),
            Arc::new(crate::provider::MockProvider::completing("done")),
            Arc::new(ToolRegistry::new()),
        )
        .with_config(config)
        .with_context_engine(Arc::new(RecordingContext { log: log.clone() }))
        .with_memory(vec![Arc::new(RecordingMemory { log: log.clone() })]);
        engine.push_user(UserMsg::new("hello"));

        engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .unwrap();
        engine.end_session().await;

        let order = log.lock().unwrap().clone();
        assert_eq!(
            order,
            vec![
                "on_model",
                "session_start",
                "switch:start",
                "recall",
                "prompt_block",
                "before_turn",
                "before_compact",
                "compact",
                "switch:compaction",
                "after_turn",
                "after_response",
                "session_end",
                "switch:end",
            ]
        );
    }
}
