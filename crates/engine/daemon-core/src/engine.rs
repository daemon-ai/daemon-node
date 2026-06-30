// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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
    HostRequestHandler, HostRequestKind, RewindAnchor, SpawnSeed, SpawnSpec, TurnSummary,
    TurnTrigger, UserMsg,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

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

/// The result of a successful [`Engine::rewind_to`]: the rewound point, the new epoch, and the tool
/// call-ids in the sealed-off tail, so the caller (the host) can drive the durable journal seal and
/// the matching workspace-checkpoint rollback (conversation-rewind spec §6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RewindOutcome {
    /// The number of conversation turns retained (turns `[0, retained_turns)` survive).
    pub retained_turns: usize,
    /// The new incarnation epoch after the rewind bump.
    pub epoch: Epoch,
    /// The `call_id`s of every tool call in the truncated tail, oldest first. The host maps these to
    /// the §12 workspace checkpoints captured before those tools ran and rolls the filesystem back to
    /// the earliest one (each checkpoint snapshots pre-mutation state, so restoring the earliest
    /// undoes every later mutation in the sealed range).
    pub dropped_call_ids: Vec<String>,
}

/// Why a [`RewindAnchor`](daemon_protocol::RewindAnchor) could not be resolved against the live
/// conversation (conversation-rewind spec §2/§5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewindError {
    /// The anchor addresses a turn outside the live conversation (out of range, or compacted away
    /// below the live floor).
    OutOfRange,
    /// A `UserTurn`/`ReplyAfter` anchor pointed at a turn that is not a user turn.
    NotAUserTurn,
}

impl std::fmt::Display for RewindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RewindError::OutOfRange => write!(f, "rewind anchor out of range"),
            RewindError::NotAUserTurn => write!(f, "rewind anchor is not a user turn"),
        }
    }
}

impl std::error::Error for RewindError {}

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
    /// The owning *identity* profile (§5.9 routed profile) this engine's §10/§11 stores are scoped
    /// under — surfaced to tools via [`TurnCx::profile`](crate::turn::TurnCx) so an `lcm_*`/`mnemosyne_*`
    /// tool resolves the same profile-rooted bank as the engine's context/memory hooks. Distinct from
    /// [`Self::profile`] (the credential profile, which mutates on a fallback hop). `None` => node default.
    subsystem_profile: Option<ProfileRef>,
    config: Config,
    /// The contained execution environment (§13) tools run in; the host injects a per-session
    /// workspace-rooted one via [`crate::EngineProfile`], else the default sandbox.
    exec: Arc<dyn ExecutionEnvironment>,
    /// The checkpoint store (§12 safety): when set, the pipeline checkpoints the workspace before a
    /// mutating tool runs (rewind via the `Checkpoint{List,Rewind}` control surface). `None` => off.
    checkpoints: Option<Arc<dyn crate::checkpoint::CheckpointStore>>,
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

/// The runtime handles a turn's early-stop finalizer needs: the §17 event sink, the cancel token for
/// the final summary call, and the folded usage so far. Bundles the shared trio threaded through
/// [`Engine::finish_with_final_summary`] and its budget/no-progress wrappers.
struct EarlyStop<'a> {
    events: &'a EventSink,
    cancel: &'a CancellationToken,
    turn_usage: UsageDelta,
}

/// The §4.3 post-turn review-cadence inputs `run_turn` accumulates over a turn and hands to
/// [`Engine::maybe_emit_reviews`]: the tool-executing rounds and whether a skill / memory tool ran.
struct ReviewSignals {
    tool_rounds: u32,
    used_skill_tool: bool,
    used_memory_tool: bool,
}

/// A durable suspension handoff: the delegated job id and its opaque worker payload. Bundles the two
/// values [`Engine::suspend`] threads onto the [`Suspension`].
struct Handoff {
    job_id: JobId,
    payload: Vec<u8>,
}

/// The pre-cloned handles [`Engine::execute_tool_batch`] runs a tool batch against without holding
/// `&mut self` across an await: the per-turn [`TurnCx`], the tool registry, the shared
/// [`TurnControl`] (boundary checks), and the §17 event sink.
struct BatchCtx<'a, 'c> {
    cx: &'a TurnCx<'c>,
    registry: &'a Arc<ToolRegistry>,
    control: &'a TurnControl,
    events: &'a EventSink,
}

impl Engine {
    /// Stash background-job completions to be applied (idempotently) before the next turn runs.
    ///
    /// Epoch fence (conversation-rewind spec §6): only completions for a job still in `waiting_for`
    /// are stashed. A `RewindTo` clears `waiting_for` (and drops the awaiting tool slots) and bumps
    /// the epoch, so a delegation completion from the abandoned tail that arrives *after* a rewind is
    /// no longer awaited and is dropped here rather than mutating the rewound conversation.
    pub fn apply_completions(&mut self, completions: Vec<Completion>) {
        for completion in completions {
            if self.snapshot.waiting_for.contains(&completion.job_id) {
                self.pending.push(completion);
            } else {
                tracing::debug!(
                    job = %completion.job_id,
                    "dropping completion for an unawaited job (fenced by rewind/epoch)"
                );
            }
        }
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

    /// Append context-only input (`AgentCommand::Observe`) into the conversation **without** opening
    /// a turn: unlike [`push_user`](Self::push_user)/[`push_steer_marker`](Self::push_steer_marker) it
    /// sets no `next_trigger` and emits no ack, so the appended chatter simply folds into the model
    /// context of the next turn the agent actually runs (the multi-party accumulation seam, §5.9).
    pub fn push_observe(&mut self, input: UserMsg) {
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

    /// Arm the next turn's [`TurnTrigger`] explicitly (a one-shot override consumed at the start of
    /// the next `run_turn`). The seam the host uses to mark a cron-fired session's turn as
    /// [`TurnTrigger::Scheduled`] (I15): the incarnation reads `SessionMeta::scheduled_job` on wake
    /// and sets it here before running, so the turn reports its scheduled origin instead of the
    /// default `User`. Has no effect if a turn is already mid-flight.
    pub fn set_next_trigger(&mut self, trigger: TurnTrigger) {
        self.next_trigger = Some(trigger);
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
        // Context-only observes that arrived mid-turn fold in as plain user context (no marker, no
        // ack, no trigger): they become part of the model context the next turn assembles.
        for input in control.drain_observe() {
            self.push_observe(input);
        }
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
        tracing::info!(
            end_reason = ?summary.end_reason,
            input_tokens = summary.usage.input_tokens,
            output_tokens = summary.usage.output_tokens,
            api_calls = summary.usage.api_calls,
            "engine.turn.finished"
        );
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
        tracing::info!(
            end_reason = ?summary.end_reason,
            input_tokens = summary.usage.input_tokens,
            output_tokens = summary.usage.output_tokens,
            api_calls = summary.usage.api_calls,
            "engine.turn.finished"
        );
        TurnOutcome::Completed(summary)
    }

    /// The current incarnation epoch.
    pub fn epoch(&self) -> Epoch {
        self.snapshot.epoch
    }

    /// The tool definitions offered to the model each turn: the core set plus either the deferrable
    /// long tail (inline) or the `tool_search` bridge (collapsed), per the §12 progressive-disclosure
    /// threshold. Resolution still spans every tool, so a collapsed tool stays callable via `tool_call`.
    fn tool_defs(&self) -> Vec<crate::tools::ToolDef> {
        self.registry
            .offered_defs(self.config.tool_search_threshold_bytes)
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
            tracing::debug!(
                attempt,
                offer_tools,
                tool_count = tools.len(),
                compacted,
                profile = %self.profile,
                "engine.model_call.attempt"
            );
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
            let step = policy.decide(&failure, attempt);
            tracing::warn!(
                attempt,
                failure_kind = failure_kind(&failure),
                recovery_step = recovery_step_kind(&step),
                "engine.model.recovery"
            );
            match step {
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
            // C6 hard last-resort cap: the context engine may return a conversation still over the
            // effective budget (compaction freed nothing, or a stateful engine kept too much). As a
            // backstop, deterministically drop the oldest turns until the estimate is within `target`
            // (or a single turn remains), so a turn never proceeds wildly over budget and blows the
            // model window. The provider-boundary §9 sequence repair fixes any tool-call/result
            // pairing broken by the drop, so this is safe even mid-conversation.
            let mut used_now = crate::context::estimate_tokens(&self.snapshot.conversation);
            while used_now > target && self.snapshot.conversation.turns.len() > 1 {
                self.snapshot.conversation.turns.remove(0);
                used_now = crate::context::estimate_tokens(&self.snapshot.conversation);
            }
            if used_now > target {
                // A single turn still exceeds the budget — cannot reduce further by dropping turns.
                // Proceed (the §8 provider-overflow recovery is the last backstop) but make the
                // unrecoverable-by-truncation case visible.
                tracing::warn!(
                    used = used_now,
                    target,
                    "context still over budget after compaction + hard truncation (single oversized turn)"
                );
            }
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
            // Decode the structured child result for its summary (the artifact refs are materialized
            // node-side by the incarnation; the engine surfaces only the text). Back-compat: the
            // decode falls back to the raw bytes as the summary, so a legacy `child:{id}` payload
            // still renders the same tool-result text.
            let summary = daemon_protocol::DelegationResult::decode(&completion.payload).summary;
            for turn in self.snapshot.conversation.turns.iter_mut() {
                if let Turn::Tool(tool_turn) = turn {
                    for (_call, result) in tool_turn.calls.iter_mut() {
                        if result.content.contains(completion.job_id.as_str()) {
                            // Deterministic value => applying the same completion twice is a no-op.
                            result.ok = true;
                            result.content = format!("completed:{}:{}", completion.job_id, summary);
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
        let span = tracing::info_span!(
            "engine.turn",
            session = %self.snapshot.session_id,
            epoch = self.snapshot.epoch.0,
            rounds_budget = self.config.max_iterations,
        );
        async {
            let resuming = !self.pending.is_empty();
            let trigger = self.next_trigger.take().unwrap_or(if resuming {
                TurnTrigger::BackgroundCompletion {
                    source: CompletionSource::Delegation(self.pending[0].job_id.clone()),
                }
            } else {
                TurnTrigger::User
            });
            tracing::info!(
                session = %self.snapshot.session_id,
                epoch = self.snapshot.epoch.0,
                resuming,
                trigger = ?trigger,
                "engine.turn.started"
            );
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
                if let Some(next) = self
                    .snapshot
                    .pending_approvals
                    .first()
                    .map(|a| a.job_id.clone())
                {
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
                // Recovery re-suspend of an already-enqueued job: the durable outbox `OR IGNORE`-dedupes
                // the re-enqueue on `(session, epoch, job_id)`, so this payload is discarded in favor of
                // the original. Pass the legacy marker to preserve prior bytes for this path.
                return Ok(self.suspend(
                    Handoff {
                        job_id,
                        payload: b"delegated-work".to_vec(),
                    },
                    events,
                    false,
                ));
            }

            // §10/§11 once-per-incarnation lifecycle hooks before the first turn's work.
            self.ensure_session_started(resuming).await;

            // §11/§10 pre-turn hooks: gather memory recall/blocks into the assembler, then measure
            // budget pressure and compact if over budget (memory.before_compact -> ctx.compact).
            let context_session = self.snapshot.session_id.clone();
            let context_epoch = self.snapshot.epoch.0;
            let context_span = tracing::debug_span!(
                "engine.context.prepare",
                session = %context_session,
                epoch = context_epoch
            );
            self.prepare_turn_context(events)
                .instrument(context_span)
                .await;

            // The in-turn ReAct loop (§4.2): build_context -> call_model -> execute_tools -> observe ->
            // call_model again, until the model returns final text (completion) or a tool delegates
            // (durable suspension). The iteration budget is the hard stop. The loop runs fully
            // in-process within one durable turn; only `Effect::Delegate` crosses the suspension boundary.
            let exec = self.exec.clone();
            let checkpoints = self.checkpoints.clone();
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
            // §4.2 no-progress guard: the signature of the previous tool round and how many consecutive
            // identical rounds we have seen. A model that keeps re-issuing the same calls and getting the
            // same results is looping; we end the turn before it burns the whole iteration budget.
            let mut last_round_sig: Option<u64> = None;
            let mut repeated_rounds: u32 = 0;

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

                let model_session = self.snapshot.session_id.clone();
                let model_epoch = self.snapshot.epoch.0;
                let model_span = tracing::info_span!(
                    "engine.model_call",
                    session = %model_session,
                    epoch = model_epoch,
                    offer_tools = true
                );
                let out = match self
                    .call_model(events, true, &cancel)
                    .instrument(model_span)
                    .await
                {
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
                    self.maybe_emit_reviews(
                        host,
                        ReviewSignals {
                            tool_rounds,
                            used_skill_tool,
                            used_memory_tool,
                        },
                    )
                    .await;
                    return Ok(self.complete(out, events, turn_usage));
                }

                let cx = TurnCx {
                    cancel: control.cancel_token(),
                    events,
                    host,
                    session_id: self.snapshot.session_id.clone(),
                    profile: self.subsystem_profile.clone(),
                    budget: self.budget,
                    exec: &*exec,
                    tool_result_budget,
                    approval_policy: effective_policy,
                    pre_approved: false,
                    checkpoints: checkpoints.as_deref(),
                };

                // Count this tool-executing round and note skill/memory tool use for the review nudges.
                tool_rounds = tool_rounds.saturating_add(1);
                for c in &out.tool_calls {
                    if c.name.starts_with("skill_manage") {
                        used_skill_tool = true;
                    } else if c.name.starts_with("mnemosyne_") {
                        used_memory_tool = true;
                    }
                }

                // execute_tools: run the model's tool batch through the §12 pipeline, collecting result
                // slots, effects, and structured detail for the rich §17 transcript views.
                let (calls, effects, interrupted) = self
                    .execute_tool_batch(
                        &out.tool_calls,
                        BatchCtx {
                            cx: &cx,
                            registry: &registry,
                            control,
                            events,
                        },
                    )
                    .await;

                // §4.2 no-progress signature of this round (the tool calls + their results), computed
                // before `calls` is moved into the recorded turn. Repeated identical rounds => looping.
                let round_sig = round_signature(&calls);

                // The single-owner applier: record the assembled tool turn, then apply any extra effects.
                self.snapshot.conversation.turns.push(Turn::Tool(ToolTurn {
                    assistant: AssistantMsg {
                        text: out.text.clone(),
                        reasoning: out.reasoning.clone(),
                    },
                    calls,
                }));
                // Route the batch's effects: persisted turns land on the conversation (after the
                // recorded tool turn, before any spawn is issued — the original ordering), the rest
                // drive suspension / fire-and-forget below.
                let PartitionedEffects {
                    persists,
                    delegated,
                    spawns,
                    awaiting,
                } = partition_tool_effects(effects);
                for turn in persists {
                    self.snapshot.conversation.turns.push(turn);
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

                if let Some((job_id, payload)) = delegated {
                    // A delegation crosses the durable boundary: notify memory of the handoff, then
                    // suspend the turn and wait for the wake.
                    self.notify_session_switch(SwitchReason::Handoff).await;
                    self.snapshot.waiting_for.push(job_id.clone());
                    return Ok(self.suspend(Handoff { job_id, payload }, events, true));
                }

                // §4.2 no-progress guard: a tool round whose calls + results are byte-identical to the
                // immediately preceding round is the model looping. Count consecutive repeats and, once
                // they reach `max_repeated_rounds`, end the turn cleanly (`NoProgress`) instead of
                // burning the rest of the iteration budget re-running the same work.
                if last_round_sig == Some(round_sig) {
                    repeated_rounds += 1;
                } else {
                    repeated_rounds = 0;
                }
                last_round_sig = Some(round_sig);
                if self.config.max_repeated_rounds > 0
                    && repeated_rounds + 1 >= self.config.max_repeated_rounds
                {
                    self.after_turn_memory().await;
                    self.context.after_response(&out.usage);
                    tracing::debug!(
                        repeated_rounds = repeated_rounds + 1,
                        "engine.react_round.no_progress"
                    );
                    return Ok(self.finish_no_progress(events, &cancel, turn_usage).await);
                }

                // §11 -> §10 post-round hooks (spec order) on the recorded tool turn, then loop — the
                // next `call_model` sees the tool results in context.
                self.after_turn_memory().await;
                self.context.after_response(&out.usage);
            }
        }
        .instrument(span)
        .await
    }

    /// Finalize a turn that hit its iteration budget: one final toolless model call to summarize,
    /// then `TurnFinished { BudgetExhausted }`. Any tool calls the model attempts on this pass are
    /// ignored — the turn is ending.
    async fn finish_budget_exhausted(
        &mut self,
        events: &EventSink,
        cancel: &CancellationToken,
        turn_usage: UsageDelta,
    ) -> TurnOutcome {
        self.finish_with_final_summary(
            EndReason::BudgetExhausted,
            EarlyStop {
                events,
                cancel,
                turn_usage,
            },
        )
        .await
    }

    /// Finalize a turn stopped early by the §4.2 no-progress guard: one final toolless summary call,
    /// then `TurnFinished { NoProgress }`. Same shape as [`Self::finish_budget_exhausted`] — a stuck
    /// loop is ended deliberately rather than left to exhaust the iteration budget.
    async fn finish_no_progress(
        &mut self,
        events: &EventSink,
        cancel: &CancellationToken,
        turn_usage: UsageDelta,
    ) -> TurnOutcome {
        self.finish_with_final_summary(
            EndReason::NoProgress,
            EarlyStop {
                events,
                cancel,
                turn_usage,
            },
        )
        .await
    }

    /// Shared early-stop finalizer: make one final **toolless** model call so the model can produce a
    /// closing message, fold its usage, and emit `TurnFinished { end_reason }`. Used by both the
    /// iteration-budget and no-progress stops.
    async fn finish_with_final_summary(
        &mut self,
        end_reason: EndReason,
        stop: EarlyStop<'_>,
    ) -> TurnOutcome {
        let EarlyStop {
            events,
            cancel,
            mut turn_usage,
        } = stop;
        let out = match self.call_model(events, false, cancel).await {
            Ok(out) => out,
            Err(f) => return self.finish_failed(f, events),
        };
        self.finalize_text(&out, events);
        turn_usage.add(&out.usage);
        let summary = TurnSummary {
            end_reason,
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
    fn complete(
        &self,
        out: ModelOutput,
        events: &EventSink,
        turn_usage: UsageDelta,
    ) -> TurnOutcome {
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
    fn suspend(&mut self, handoff: Handoff, events: &EventSink, bump_epoch: bool) -> TurnOutcome {
        let Handoff { job_id, payload } = handoff;
        if bump_epoch {
            self.snapshot.epoch = self.snapshot.epoch.next();
        }
        let summary = TurnSummary::ended(EndReason::Suspended);
        events.emit(|seq| AgentEvent::TurnFinished { seq, summary });
        tracing::info!(
            job_id = %job_id,
            epoch = self.snapshot.epoch.0,
            suspend_kind = "delegation",
            "engine.turn.suspended"
        );
        TurnOutcome::Suspended(Suspension {
            job_id,
            epoch: self.snapshot.epoch,
            payload,
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
        tracing::info!(
            job_id = %job_id,
            epoch = self.snapshot.epoch.0,
            suspend_kind = "approval",
            "engine.turn.suspended"
        );
        TurnOutcome::Suspended(Suspension {
            job_id,
            epoch: self.snapshot.epoch,
            payload: APPROVAL_SUSPEND_PAYLOAD.to_vec(),
        })
    }

    /// Run the model's tool batch through the §12 pipeline, collecting result slots, effects, and
    /// structured detail for the rich §17 transcript views; returns `(calls, effects, interrupted)`.
    ///
    /// A batch runs **concurrently** only when it has more than one call and *every* call resolves to
    /// a tool that declares itself [`ToolConcurrency::Parallel`](crate::tools::ToolConcurrency)
    /// (all-or-nothing; any exclusive/mutating call serializes the batch). Strict §17 event ordering
    /// is preserved: per-call `ToolStarted`→`ToolFinished` in call order (the parallel branch emits
    /// every start in order, runs the batch, then drains every finish in order). The read-only steer
    /// boundary is evaluated once after a parallel batch settles, but after *each* serial tool (an
    /// interrupt stops further serial execution). Takes `&self` and the pre-cloned `cx`/`registry`
    /// handles (so no `&mut self` is held across a tool await — the `resolve_approvals` pattern).
    async fn execute_tool_batch(
        &self,
        tool_calls: &[crate::conversation::ToolCall],
        batch: BatchCtx<'_, '_>,
    ) -> (
        Vec<(
            crate::conversation::ToolCall,
            crate::conversation::ToolResult,
        )>,
        Vec<Effect>,
        bool,
    ) {
        let BatchCtx {
            cx,
            registry,
            control,
            events,
        } = batch;
        let mut calls = Vec::new();
        let mut effects: Vec<Effect> = Vec::new();
        let mut interrupted = false;

        let parallel = tool_calls.len() > 1
            && tool_calls.iter().all(|c| {
                registry
                    .get(&c.name)
                    .map(|t| t.concurrency() == crate::tools::ToolConcurrency::Parallel)
                    .unwrap_or(false)
            });

        if parallel {
            // Emit all starts in call order, run the batch concurrently, then drain results in
            // call order. Read-only parallel tools have no ordered side effects, so the boundary
            // is evaluated once after the whole batch settles.
            for call in tool_calls {
                let view = tool_call_view(call);
                events.emit(|seq| AgentEvent::ToolStarted { seq, call: view });
            }
            let outcomes = futures::future::join_all(
                tool_calls
                    .iter()
                    .map(|call| async { (call.clone(), run_tool(call, registry, cx).await) }),
            )
            .await;
            for (call, outcome) in outcomes {
                let result_view = tool_result_view(&outcome);
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
            for call in tool_calls {
                let view = tool_call_view(call);
                events.emit(|seq| AgentEvent::ToolStarted { seq, call: view });
                let outcome = run_tool(call, registry, cx).await;
                let result_view = tool_result_view(&outcome);
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

        (calls, effects, interrupted)
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
        let checkpoints = self.checkpoints.clone();
        let registry = self.registry.clone();
        let budget = self.budget;
        let tool_result_budget = self.config.tool_result_budget;
        let policy = self.effective_policy();
        let session_id = self.snapshot.session_id.clone();
        let subsystem_profile = self.subsystem_profile.clone();
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
                            profile: subsystem_profile.clone(),
                            budget,
                            exec: &*exec,
                            tool_result_budget,
                            approval_policy: policy,
                            pre_approved: true,
                            checkpoints: checkpoints.as_deref(),
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
    async fn maybe_emit_reviews(&mut self, host: &dyn HostRequestHandler, signals: ReviewSignals) {
        let ReviewSignals {
            tool_rounds,
            used_skill_tool,
            used_memory_tool,
        } = signals;
        let mut spawns: Vec<SpawnSpec> = Vec::new();

        // Skill review: count this turn's tool iterations, but a `skill_manage` use this turn resets
        // the cadence (the agent just curated skills — no nudge needed).
        self.snapshot.iters_since_skill =
            self.snapshot.iters_since_skill.saturating_add(tool_rounds);
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

mod builder;
mod rewind;
mod views;
use views::{
    failure_kind, partition_tool_effects, recovery_step_kind, round_signature, tool_call_view,
    tool_result_view, PartitionedEffects,
};

#[cfg(test)]
mod support;

#[cfg(test)]
mod tests;
