//! The model provider port (§7) and the deterministic [`MockProvider`]/[`ScriptedProvider`].
//!
//! The engine talks to a model through this trait, not a concrete client, so providers are
//! swappable and standalone-embeddable. The trait now carries streaming ([`Provider::stream`] +
//! [`StreamEvent`]) with a `chat()`-adapting default and the §8 [`Failure`] taxonomy + [`Recovery`]
//! mapping consumed by [`crate::recovery`]. The in-tree providers stay deterministic (no network);
//! real networked clients live in the sibling `daemon-providers` crate.

use crate::conversation::{Conversation, ToolCall, Turn};
use crate::profile::ProviderBuilder;
use crate::tools::ToolDef;
use daemon_common::{ProfileRef, UsageDelta};
use futures::stream::{self, BoxStream};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

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
    /// The tools offered this turn, each with its JSON-Schema (the §12 registry's
    /// [`ToolDef`](crate::tools::ToolDef)s). A provider that supports native tools sends the schema;
    /// a name-only consumer can use [`Request::tool_names`].
    pub tools: Vec<ToolDef>,
    /// The bearer credential for the call, threaded from the acquired capability lease
    /// ([`daemon_common::CapabilityLease::secret`]). `None` for the deterministic in-tree providers
    /// (which ignore it); a real networked provider sends it as the `Authorization` bearer. Treat as
    /// a secret — never log it.
    pub auth: Option<String>,
    /// An optional grammar constraint bounding the model's output to a formal language (e.g. the
    /// MeTTa "draft" path that constrains generation to the symbolic-coprocessor grammar). `None` =
    /// unconstrained. A provider that cannot constrain generation ignores it.
    pub constraint: Option<GrammarConstraint>,
    /// Whether to mark the system prompt as a prompt-cache breakpoint (§ prompt caching). Set by the
    /// engine's cache policy; a provider that supports prefix caching (e.g. Anthropic via
    /// `cache_control`) caches the tools+system prefix, and a provider without it ignores the hint.
    pub cache_system: bool,
}

/// An engine-agnostic grammar constraint carried on a [`Request`]. It holds both grammar dialects so
/// any local engine can pick the one it supports (mistral.rs => [`GrammarConstraint::lark`], llama
/// => [`GrammarConstraint::gbnf`]); networked providers that cannot constrain output ignore it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrammarConstraint {
    /// A Lark grammar (mistral.rs / llguidance), if available.
    pub lark: Option<String>,
    /// A GBNF grammar (llama.cpp, root rule `root`), if available.
    pub gbnf: Option<String>,
}

impl Request {
    /// Whether any message in the request is a tool result (a resolved tool turn).
    pub fn has_tool_result(&self) -> bool {
        self.messages.iter().any(|m| m.role == "tool")
    }

    /// The names of the offered tools (the valid set §9 tool-name repair resolves against).
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    /// Bound this request's output to `constraint` (e.g. the MeTTa "draft" grammar). A provider that
    /// cannot constrain generation ignores it.
    pub fn with_constraint(mut self, constraint: GrammarConstraint) -> Self {
        self.constraint = Some(constraint);
        self
    }
}

/// One flattened message in a [`Request`].
///
/// Carries the native tool-call linkage so a provider can round-trip a tool exchange faithfully: an
/// `assistant` message that called tools fills [`RequestMsg::tool_calls`], and the matching `tool`
/// result message fills [`RequestMsg::tool_call_id`]. Name-only providers can ignore both.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RequestMsg {
    /// The role: `user`, `assistant`, or `tool`.
    pub role: String,
    /// The message content (assistant/user text, or the tool result payload).
    pub content: String,
    /// For an `assistant` message: the tool calls it emitted (native round-trip).
    pub tool_calls: Vec<ToolCall>,
    /// For a `tool` message: the id of the call this result answers (native round-trip).
    pub tool_call_id: Option<String>,
    /// A prompt-cache breakpoint marker (§ prompt caching): when set, the stable prefix up to and
    /// including this message should be cached. The engine's cache policy marks the last message of
    /// the request so the growing conversation reuses the cached prefix across turns; a provider
    /// without prefix caching ignores it.
    pub cache_breakpoint: bool,
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

/// One event in a streamed model response (§7). Real providers emit incremental
/// [`StreamEvent::TextDelta`]/[`StreamEvent::ReasoningDelta`] as SSE chunks arrive and a terminal
/// [`StreamEvent::Done`] carrying the assembled canonical [`ModelOutput`]; deterministic providers
/// emit a single `Done` via the [`Provider::stream`] default.
#[derive(Clone, Debug)]
pub enum StreamEvent {
    /// Incremental assistant text.
    TextDelta(String),
    /// Incremental reasoning text (the separate reasoning channel).
    ReasoningDelta(String),
    /// Incremental usage accounting (folded into the running total).
    Usage(UsageDelta),
    /// The terminal canonical output (text/reasoning/tool-calls/usage assembled).
    Done(ModelOutput),
}

/// What recovery action a [`Failure`] suggests (§8). The recovery middleware
/// ([`crate::recovery`]) maps failures to one of these and drives retry/rotate/compact/fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Recovery {
    /// Retry after an optional backoff (honors a server `Retry-After`).
    Retry {
        /// The minimum delay before retrying, if the provider specified one.
        after: Option<Duration>,
    },
    /// Rotate the credential and retry (quota/auth on a poolable key).
    Rotate,
    /// The request exceeded the context window — compact then retry once (§8 -> §10 tie-in).
    Compact,
    /// Hop to the single fallback profile (the failure is not retryable on this profile).
    Fallback,
    /// Give up — the turn ends `Failed`.
    Abort,
}

/// A model/provider failure (§8 taxonomy). The legacy [`Failure::Provider`]/[`Failure::Rotatable`]/
/// [`Failure::Other`] variants are retained for the in-tree call sites; the networked providers
/// classify HTTP responses into the precise variants via
/// [`classify_api_error`](crate::recovery::classify_api_error).
#[derive(Debug, thiserror::Error)]
pub enum Failure {
    /// The provider itself failed (unclassified).
    #[error("provider: {0}")]
    Provider(String),
    /// A rotatable provider failure (quota/rate-limit/auth on a poolable key): mark the credential
    /// and retry on a rotated one (`credential_pool.py` `should_rotate`).
    #[error("rotatable: {0}")]
    Rotatable(String),
    /// Rate limited (HTTP 429) — back off, honoring `Retry-After` when present.
    #[error("rate limited")]
    RateLimit {
        /// The server-advised minimum delay before retrying, if any.
        retry_after: Option<Duration>,
        /// A human-readable detail.
        message: String,
    },
    /// A billing/quota-exhausted condition (HTTP 402) — not retryable on this credential.
    #[error("billing: {0}")]
    Billing(String),
    /// An authentication/authorization failure (HTTP 401/403) — rotate the credential.
    #[error("auth: {0}")]
    Auth(String),
    /// The request exceeded the model context window — compact and retry once.
    #[error("context overflow: {0}")]
    ContextOverflow(String),
    /// The request payload was too large (HTTP 413) — compact and retry once.
    #[error("payload too large: {0}")]
    PayloadTooLarge(String),
    /// The request/response tripped a content policy filter — not retryable as-is.
    #[error("content policy: {0}")]
    ContentPolicy(String),
    /// The provider returned a malformed/unparseable response — retry once to re-elicit.
    #[error("format error: {0}")]
    FormatError(String),
    /// A transient transport error (timeout, reset, hung stream) — retry with backoff.
    #[error("transient transport: {0}")]
    TransientTransport(String),
    /// The provider is overloaded (HTTP 503/529) — retry with backoff.
    #[error("provider overloaded: {0}")]
    ProviderOverloaded(String),
    /// An unrecoverable provider error — abort the turn.
    #[error("fatal: {0}")]
    Fatal(String),
    /// The turn was cancelled cooperatively.
    #[error("cancelled")]
    Cancelled,
    /// Any other engine failure.
    #[error("{0}")]
    Other(String),
}

impl Failure {
    /// Whether this failure should trigger a credential rotation + retry. Quota/auth-class failures
    /// rotate; `RateLimit` is also rotatable (a fresh key may have headroom).
    pub fn is_rotatable(&self) -> bool {
        matches!(
            self,
            Failure::Rotatable(_)
                | Failure::Auth(_)
                | Failure::Billing(_)
                | Failure::RateLimit { .. }
        )
    }

    /// The recovery action this failure suggests (§8). Exhaustive over the taxonomy so a new variant
    /// forces a deliberate recovery decision.
    pub fn recovery(&self) -> Recovery {
        match self {
            Failure::RateLimit { retry_after, .. } => Recovery::Retry {
                after: *retry_after,
            },
            Failure::TransientTransport(_) | Failure::ProviderOverloaded(_) => {
                Recovery::Retry { after: None }
            }
            // A malformed response: one bounded retry can re-elicit a clean parse.
            Failure::FormatError(_) => Recovery::Retry { after: None },
            Failure::Rotatable(_) | Failure::Auth(_) => Recovery::Rotate,
            // Billing/content-policy can't clear on the same profile: hop to the fallback profile.
            Failure::Billing(_) | Failure::ContentPolicy(_) => Recovery::Fallback,
            Failure::ContextOverflow(_) | Failure::PayloadTooLarge(_) => Recovery::Compact,
            Failure::Provider(_) | Failure::Other(_) | Failure::Fatal(_) | Failure::Cancelled => {
                Recovery::Abort
            }
        }
    }
}

/// The model provider port (§7).
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Declared capabilities.
    fn capabilities(&self) -> Capabilities;
    /// Run a (non-streaming) chat completion.
    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure>;
    /// Stream a chat completion as [`StreamEvent`]s, terminating with [`StreamEvent::Done`].
    ///
    /// The provided default adapts [`Provider::chat`] into a single terminal `Done` event, so a
    /// non-streaming provider (the deterministic [`MockProvider`]/[`ScriptedProvider`]) needs no
    /// extra code; a real streaming provider overrides this to forward SSE deltas as they arrive.
    fn stream(&self, req: Request) -> BoxStream<'_, Result<StreamEvent, Failure>> {
        Box::pin(stream::once(async move {
            self.chat(req).await.map(StreamEvent::Done)
        }))
    }
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

/// Flatten a conversation into a [`Request`] (the `build_context` phase). A tool turn becomes an
/// `assistant` message carrying its native [`ToolCall`]s plus one `tool` message per result (tagged
/// with its `call_id`), so a native provider round-trips the exchange faithfully.
pub fn build_context(conv: &Conversation, tools: &[ToolDef]) -> Request {
    let mut messages = Vec::new();
    for turn in &conv.turns {
        match turn {
            Turn::User(u) => messages.push(RequestMsg {
                role: "user".into(),
                content: u.text.clone(),
                ..Default::default()
            }),
            Turn::Assistant(a) => messages.push(RequestMsg {
                role: "assistant".into(),
                content: a.text.clone(),
                ..Default::default()
            }),
            Turn::Tool(t) => {
                messages.push(RequestMsg {
                    role: "assistant".into(),
                    content: t.assistant.text.clone(),
                    tool_calls: t.calls.iter().map(|(call, _)| call.clone()).collect(),
                    ..Default::default()
                });
                for (_call, result) in &t.calls {
                    messages.push(RequestMsg {
                        role: "tool".into(),
                        content: result.content.clone(),
                        tool_call_id: Some(result.call_id.clone()),
                        ..Default::default()
                    });
                }
            }
        }
    }
    // Repair the flattened wire sequence (§9): Turn-granular compaction can drop a leading user turn
    // or leave a suspended tool call unanswered, so enforce the provider structural contract here
    // (leading-user, tool pairing, no empty blocks). No-op for a well-formed sequence.
    let messages = crate::repair::repair_message_sequence(messages);
    let mut req = Request {
        system: conv.system.text.clone(),
        messages,
        tools: tools.to_vec(),
        auth: None,
        constraint: None,
        cache_system: false,
    };
    mark_cache_breakpoints(&mut req);
    req
}

/// Place prompt-cache breakpoints on the request's stable prefix (§ prompt caching).
///
/// Two breakpoints, mirroring Anthropic's incremental multi-turn recommendation and hermes'
/// `prompt_caching` policy: (1) the **tools+system** prefix (the largest stable block, unchanged
/// turn to turn), and (2) the **last message** of the request, so the entire conversation prefix is
/// cached and the *next* turn reads it as a hit before appending. Providers without prefix caching
/// ignore both markers, so this is always safe to apply.
pub fn mark_cache_breakpoints(req: &mut Request) {
    req.cache_system = !req.system.is_empty();
    if let Some(last) = req.messages.last_mut() {
        last.cache_breakpoint = true;
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
            ..Default::default()
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
            ..Default::default()
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

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::conversation::{AssistantMsg, Conversation, SystemPrompt};
    use daemon_protocol::UserMsg;

    #[test]
    fn build_context_marks_system_and_last_message() {
        let mut conv = Conversation::new(SystemPrompt::new("a stable system prompt"));
        conv.push_user(UserMsg::new("hello"));
        conv.push_assistant(AssistantMsg::text("hi there"));
        let req = build_context(&conv, &[]);

        assert!(req.cache_system, "a non-empty system is a cache breakpoint");
        assert!(
            req.messages.last().unwrap().cache_breakpoint,
            "the last message anchors the conversation-prefix breakpoint"
        );
        assert_eq!(
            req.messages.iter().filter(|m| m.cache_breakpoint).count(),
            1,
            "exactly one message-level breakpoint (the last)"
        );
    }

    #[test]
    fn empty_system_is_not_a_cache_breakpoint() {
        let mut conv = Conversation::new(SystemPrompt::new(""));
        conv.push_user(UserMsg::new("hello"));
        let req = build_context(&conv, &[]);
        assert!(!req.cache_system);
        // The message-prefix breakpoint still applies even without a system.
        assert!(req.messages.last().unwrap().cache_breakpoint);
    }

    #[test]
    fn no_messages_yields_no_message_breakpoint() {
        let conv = Conversation::new(SystemPrompt::new("sys"));
        let req = build_context(&conv, &[]);
        assert!(req.cache_system);
        assert!(req.messages.is_empty());
    }
}
