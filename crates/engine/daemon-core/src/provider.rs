//! The model provider port (§7) and a deterministic [`MockProvider`].
//!
//! The engine talks to a model through this trait, not a concrete client, so providers are
//! swappable and standalone-embeddable. Phase 3 ships only the [`MockProvider`]; real provider
//! clients (and streaming) arrive later.

use crate::conversation::{Conversation, ToolCall, Turn};
use crate::profile::ProviderBuilder;
use daemon_common::{ProfileRef, UsageDelta};
use std::collections::HashMap;
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
    /// A rotatable provider failure (quota/rate-limit/auth, e.g. HTTP 429/402/401): the engine
    /// should mark the credential and retry on a rotated one (`credential_pool.py` `should_rotate`).
    #[error("rotatable: {0}")]
    Rotatable(String),
    /// The turn was cancelled cooperatively.
    #[error("cancelled")]
    Cancelled,
    /// Any other engine failure.
    #[error("{0}")]
    Other(String),
}

impl Failure {
    /// Whether this failure should trigger a credential rotation + retry.
    pub fn is_rotatable(&self) -> bool {
        matches!(self, Failure::Rotatable(_))
    }
}

/// The model provider port (§7).
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Declared capabilities.
    fn capabilities(&self) -> Capabilities;
    /// Run a (non-streaming) chat completion.
    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure>;
}

/// A name -> [`ProviderBuilder`] map with a fallback default — the provider *selection* seam.
///
/// Provider implementations are a `daemon-core` port; *which* provider a profile resolves to is a
/// host/binary policy. The registry makes that policy a one-line registration: a real networked
/// provider drops in by `register("openai", ...)` (or `set_default`) without touching the engine or
/// the construction sites. Phase 9 ships [`MockProvider`] as the default; no networked provider yet.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    builders: HashMap<String, ProviderBuilder>,
    default: Option<ProviderBuilder>,
}

impl ProviderRegistry {
    /// An empty registry (no default; [`ProviderRegistry::builder_for`] returns `None` until one is
    /// registered or set as default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the provider builder a given profile name resolves to.
    pub fn register(&mut self, profile: impl Into<String>, builder: ProviderBuilder) {
        self.builders.insert(profile.into(), builder);
    }

    /// Set the fallback builder used when a profile has no explicit registration.
    pub fn set_default(&mut self, builder: ProviderBuilder) {
        self.default = Some(builder);
    }

    /// Resolve the builder for `profile`: its explicit registration, else the default, else `None`.
    pub fn builder_for(&self, profile: &ProfileRef) -> Option<ProviderBuilder> {
        self.builders
            .get(profile.as_str())
            .cloned()
            .or_else(|| self.default.clone())
    }
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

/// One scripted model round for a [`ScriptedProvider`].
#[derive(Clone, Debug)]
pub enum ScriptStep {
    /// Emit a single tool call with the given name and (canonical) argument payload.
    Call {
        /// The tool name to invoke.
        name: String,
        /// The argument payload (e.g. a JSON object for the fs/shell tools).
        args: String,
    },
    /// Emit several tool calls in one round (a parallel batch the engine runs in order).
    Calls(Vec<(String, String)>),
    /// Emit final assistant text and no tool calls (completes the turn).
    Final(String),
}

/// A deterministic provider that replays a fixed sequence of tool-call/final rounds — the seam-only
/// stand-in for a real model that drives the multi-round ReAct loop in tests (no network/keys).
///
/// Each `chat` returns the next [`ScriptStep`]. Once the script is exhausted it returns `final_text`
/// (completing the turn), unless constructed with [`ScriptedProvider::looping`], which repeats one
/// step forever — useful for exercising the iteration-budget hard stop.
pub struct ScriptedProvider {
    steps: Vec<ScriptStep>,
    repeat: Option<ScriptStep>,
    final_text: String,
    calls: AtomicU64,
}

impl ScriptedProvider {
    /// A provider that replays `steps` in order, then completes with `final_text`.
    pub fn new(steps: Vec<ScriptStep>, final_text: impl Into<String>) -> Self {
        Self {
            steps,
            repeat: None,
            final_text: final_text.into(),
            calls: AtomicU64::new(0),
        }
    }

    /// A provider that emits `step` on *every* round forever — never completes on its own, so the
    /// engine's iteration budget is what ends the turn (`BudgetExhausted`).
    pub fn looping(step: ScriptStep) -> Self {
        Self {
            steps: Vec::new(),
            repeat: Some(step),
            final_text: String::new(),
            calls: AtomicU64::new(0),
        }
    }

    /// How many model rounds this provider has served (test observability).
    pub fn call_count(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    fn output(&self, step: &ScriptStep, n: u64, usage: UsageDelta) -> ModelOutput {
        match step {
            ScriptStep::Call { name, args } => ModelOutput {
                text: String::new(),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    call_id: format!("call-{n}"),
                    name: name.clone(),
                    args: args.clone(),
                }],
                usage,
            },
            ScriptStep::Calls(list) => ModelOutput {
                text: String::new(),
                reasoning: None,
                tool_calls: list
                    .iter()
                    .enumerate()
                    .map(|(i, (name, args))| ToolCall {
                        call_id: format!("call-{n}-{i}"),
                        name: name.clone(),
                        args: args.clone(),
                    })
                    .collect(),
                usage,
            },
            ScriptStep::Final(text) => ModelOutput {
                text: text.clone(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage,
            },
        }
    }
}

#[async_trait::async_trait]
impl Provider for ScriptedProvider {
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
        let usage = UsageDelta {
            input_tokens: 8,
            output_tokens: 4,
            api_calls: 1,
        };
        let step = self
            .steps
            .get(n as usize)
            .or(self.repeat.as_ref())
            .cloned()
            .unwrap_or_else(|| ScriptStep::Final(self.final_text.clone()));
        Ok(self.output(&step, n, usage))
    }
}
