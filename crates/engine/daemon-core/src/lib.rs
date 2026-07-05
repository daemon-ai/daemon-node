// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-core` — the agent engine (the "brain").
//!
//! The single-owner agent actor (§4.1) that drives turns over a typed [`Conversation`] (§5),
//! producing the durable [`Snapshot`] (lifecycle §2) at each phase boundary. It composes the turn
//! phases (`build_context` → `call_model` over a [`Provider`] → `execute_tools` over the §12 tool
//! pipeline → finalize) and applies their [`Effect`]s through a single-owner applier, which makes
//! suspension a deterministic boundary the durable substrate can checkpoint and resume.
//!
//! It speaks the §17 host protocol (`daemon-protocol`) and is intentionally unaware of
//! `daemon-supervision` and the durable substrate — the host adapts the management protocol and
//! bridges the activation seam on its behalf. The turn now drives a real **in-turn ReAct loop**
//! (model → tools → model until final text) guarded by the §20 iteration budget, executing tools
//! through a §13 [`ExecutionEnvironment`] (the in-core [`LocalEnvironment`]); the provider stays
//! deterministic ([`MockProvider`] / [`ScriptedProvider`]) this phase, with real model I/O,
//! compaction, memory, and LSP arriving later.
//!
//! See `crates/engine/daemon-core/docs/` for the engine spec family.

#![forbid(unsafe_code)]

pub mod actor;
pub mod approval;
pub mod checkpoint;
pub mod command;
pub mod config;
pub mod context;
pub mod control;
pub mod conversation;
pub mod credentials;
pub mod embed;
pub mod engine;
pub mod events;
pub mod exec;
#[cfg(feature = "otel")]
pub mod genai_telemetry;
pub mod guardrail;
pub mod memory;
pub mod profile;
pub mod provider;
pub mod recovery;
pub mod repair;
pub mod safety;
pub mod snapshot;
pub mod tool_pipeline;
pub mod tools;
pub mod turn;

pub use actor::{spawn_agent_session, AgentHandle};
pub use approval::{is_sensitive_path, ApprovalPolicy, Decision};
pub use checkpoint::{CheckpointKind, CheckpointRecord, CheckpointStore, LocalCheckpointStore};
pub use command::{
    CommandAccess, CommandCx, CommandError, CommandInvocation, CommandOutput, CommandProvider,
    CommandProviderHandle, CommandScope, CommandSpec, NoCommands,
};
pub use config::Config;
pub use context::{
    estimate_tokens, BudgetedContextEngine, ContextEngine, ContextStrategy, ModelInfo, Pressure,
    PromptAssembler, StablePromptSource,
};
pub use control::{SteerReq, TurnControl};
pub use conversation::{
    AssistantMsg, Conversation, SystemPrompt, ToolCall, ToolResult, ToolTurn, Turn, UserMsg,
};
pub use credentials::{CredentialProvider, EmbeddedCredentialPool};
pub use embed::{cosine, EmbeddingProvider, MockEmbedder};
pub use engine::{
    Completion, Engine, RewindError, RewindOutcome, Suspension, TurnOutcome,
    APPROVAL_SUSPEND_PAYLOAD,
};
pub use events::{EventSink, SessionLog};
pub use exec::{contain, Command, ExecCx, ExecResult, ExecutionEnvironment, LocalEnvironment};
#[cfg(feature = "otel")]
pub use genai_telemetry::set_genai_capture;
pub use memory::{
    FileMemory, MemoryProvider, PromptBlock, RecallQuery, RecalledBlock, SwitchReason,
};
pub use profile::{
    ContextEngineBuilder, CredentialBuilder, EngineProfile, ExecEnvBuilder, MemoryBuilder,
    ProviderBuilder,
};
pub use provider::{
    build_context, Capabilities, Failure, MockProvider, ModelOutput, Provider, ProviderRegistry,
    Recovery, Request, RequestImage, RequestMsg, RequestParams, ResponseMeta, ScriptStep,
    ScriptedProvider, StreamEvent, ToolCallFormat, UnconfiguredProvider,
};
pub use recovery::{classify_api_error, drive_model_call, ModelCallPolicy, RecoveryStep};
pub use repair::{
    repair_tool_args, repair_tool_call, repair_tool_name, sanitize_tool_error, scrub_content,
    wrap_untrusted_tool_result, ArgRepair, NameRepairError, ScrubChunk, StreamingThinkScrubber,
};
pub use safety::{check_url, CheckedUrl, UrlReject};
pub use snapshot::{PendingApproval, ProcHandle, References, Snapshot, ToolBinding};
pub use tool_pipeline::run_tool;
pub use tools::{
    DelegateTool, Tool, ToolConcurrency, ToolDef, ToolOutcome, ToolProvider, ToolProviderError,
    ToolRegistry,
};
pub use turn::{approve_command, approve_path, Effect, Gate, TurnCx};
