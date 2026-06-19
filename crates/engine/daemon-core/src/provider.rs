//! The model provider port (§7) and a deterministic [`MockProvider`].
//!
//! The engine talks to a model through this trait, not a concrete client, so providers are
//! swappable and standalone-embeddable. Phase 3 ships only the [`MockProvider`]; real provider
//! clients (and streaming) arrive later.

use crate::conversation::{Conversation, ToolCall, Turn};
use daemon_common::UsageDelta;
use std::sync::atomic::{AtomicU64, Ordering};

/// How a model expects tool calls to be encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// Native tool-calling.
    Native,
    /// Anthropic tool-use blocks.
    AnthropicToolUse,
    /// Hermes-style XML.
    HermesXml,
}

/// Declared model capabilities (§7).
#[derive(Clone, Copy, Debug)]
pub struct Capabilities {
    /// Whether the model supports native tool calls.
    pub supports_native_tools: bool,
    /// Whether the model supports streaming.
    pub supports_streaming: bool,
    /// The tool-call wire format.
    pub tool_call_format: ToolCallFormat,
    /// The maximum context window, if known.
    pub max_context: Option<u32>,
}

/// The flattened request the engine hands a provider (built by `build_context`).
#[derive(Clone, Debug, Default)]
pub struct Request {
    /// The system prompt text.
    pub system: String,
    /// The flattened conversation messages.
    pub messages: Vec<RequestMsg>,
    /// The names of the tools offered this turn.
    pub tools: Vec<String>,
}

impl Request {
    /// Whether any message in the request is a tool result (a resolved tool turn).
    pub fn has_tool_result(&self) -> bool {
        self.messages.iter().any(|m| m.role == "tool")
    }
}

/// One flattened message in a [`Request`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestMsg {
    /// The role: `user`, `assistant`, or `tool`.
    pub role: String,
    /// The message content.
    pub content: String,
}

/// What a model produced for one turn (§4.4).
#[derive(Clone, Debug, Default)]
pub struct ModelOutput {
    /// The assistant text.
    pub text: String,
    /// Optional reasoning.
    pub reasoning: Option<String>,
    /// Any tool calls, already decoded to the canonical type.
    pub tool_calls: Vec<ToolCall>,
    /// Usage accrued by this model call.
    pub usage: UsageDelta,
}

/// A model/provider failure.
#[derive(Debug, thiserror::Error)]
pub enum Failure {
    /// The provider itself failed.
    #[error("provider: {0}")]
    Provider(String),
    /// The turn was cancelled cooperatively.
    #[error("cancelled")]
    Cancelled,
    /// Any other engine failure.
    #[error("{0}")]
    Other(String),
}

/// The model provider port (§7).
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Declared capabilities.
    fn capabilities(&self) -> Capabilities;
    /// Run a (non-streaming) chat completion.
    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure>;
}

/// Flatten a conversation into a [`Request`] (the `build_context` phase, minimal form).
pub fn build_context(conv: &Conversation, tools: &[String]) -> Request {
    let mut messages = Vec::new();
    for turn in &conv.turns {
        match turn {
            Turn::User(u) => messages.push(RequestMsg {
                role: "user".into(),
                content: u.text.clone(),
            }),
            Turn::Assistant(a) => messages.push(RequestMsg {
                role: "assistant".into(),
                content: a.text.clone(),
            }),
            Turn::Tool(t) => {
                messages.push(RequestMsg {
                    role: "assistant".into(),
                    content: t.assistant.text.clone(),
                });
                for (_call, result) in &t.calls {
                    messages.push(RequestMsg {
                        role: "tool".into(),
                        content: result.content.clone(),
                    });
                }
            }
        }
    }
    Request {
        system: conv.system.text.clone(),
        messages,
        tools: tools.to_vec(),
    }
}

/// The scripted behaviour of a [`MockProvider`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Behaviour {
    /// Always finish in one turn with final text.
    Completing,
    /// Call the `delegate` tool until a tool result is present, then finish with text.
    Delegating,
}

/// A deterministic provider for tests and the phase-3 substrate (no real model I/O).
///
/// - [`MockProvider::completing`] always returns final text and no tool calls — a single-turn
///   completion (the §9 round-trip happy path).
/// - [`MockProvider::delegating`] returns a `delegate` tool call on a fresh context and final text
///   once a tool result is present — reproducing the durable "delegate → suspend → resume →
///   complete" cycle the substrate conformance tests rely on.
pub struct MockProvider {
    behaviour: Behaviour,
    final_text: String,
    delegate_tool: String,
    calls: AtomicU64,
}

impl MockProvider {
    /// A provider that finishes every turn in one model call.
    pub fn completing(final_text: impl Into<String>) -> Self {
        Self {
            behaviour: Behaviour::Completing,
            final_text: final_text.into(),
            delegate_tool: "delegate".into(),
            calls: AtomicU64::new(0),
        }
    }

    /// A provider that delegates once (via the named tool) and then completes.
    pub fn delegating(delegate_tool: impl Into<String>, final_text: impl Into<String>) -> Self {
        Self {
            behaviour: Behaviour::Delegating,
            final_text: final_text.into(),
            delegate_tool: delegate_tool.into(),
            calls: AtomicU64::new(0),
        }
    }

    /// How many model calls this provider has served (test observability).
    pub fn call_count(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl Provider for MockProvider {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: false,
            tool_call_format: ToolCallFormat::Native,
            max_context: Some(8192),
        }
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        let usage = UsageDelta {
            input_tokens: 8,
            output_tokens: 4,
            api_calls: 1,
        };
        let wants_tool = matches!(self.behaviour, Behaviour::Delegating) && !req.has_tool_result();
        if wants_tool {
            Ok(ModelOutput {
                text: "delegating background work".into(),
                reasoning: Some("the request needs background work".into()),
                tool_calls: vec![ToolCall {
                    call_id: format!("call-{n}"),
                    name: self.delegate_tool.clone(),
                    args: "{}".into(),
                }],
                usage,
            })
        } else {
            Ok(ModelOutput {
                text: self.final_text.clone(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage,
            })
        }
    }
}
