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
//! bridges the activation seam on its behalf. Phase 3 ships a deterministic [`MockProvider`] and a
//! single [`DelegateTool`]; real model I/O, compaction, memory, and LSP arrive later.
//!
//! See `crates/engine/daemon-core/docs/` for the engine spec family.

#![forbid(unsafe_code)]

pub mod actor;
pub mod conversation;
pub mod credentials;
pub mod engine;
pub mod events;
pub mod provider;
pub mod snapshot;
pub mod tool_pipeline;
pub mod tools;
pub mod turn;

pub use actor::{spawn_agent_session, AgentHandle};
pub use conversation::{
    AssistantMsg, Conversation, SystemPrompt, ToolCall, ToolResult, ToolTurn, Turn,
};
pub use credentials::{CredentialProvider, EmbeddedCredentialPool};
pub use engine::{Completion, Engine, Suspension, TurnOutcome};
pub use events::EventSink;
pub use provider::{
    build_context, Capabilities, Failure, MockProvider, ModelOutput, Provider, Request, RequestMsg,
    ToolCallFormat,
};
pub use snapshot::{ProcHandle, References, Snapshot, ToolBinding};
pub use tool_pipeline::run_tool;
pub use tools::{DelegateTool, Tool, ToolDef, ToolOutcome, ToolRegistry};
pub use turn::{Effect, TurnCx};
