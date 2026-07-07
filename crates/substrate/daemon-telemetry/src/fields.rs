// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Stable field and span names for operational tracing.
//!
//! These constants keep log pipelines and tests from depending on ad-hoc spelling at each call
//! site. The verifiable journal remains the audit source of truth; these names are for runtime
//! `tracing` spans/events.

/// The current [`daemon_common::TraceId`] rendered as fixed-width hex.
pub const TRACE_ID: &str = "trace_id";
/// The broad category of work represented by a span.
pub const SPAN_KIND: &str = "span.kind";
/// Durable session id.
pub const SESSION: &str = "session";
/// Supervision/unit id.
pub const UNIT: &str = "unit";
/// Request/correlation id on live protocols.
pub const REQ_ID: &str = "req_id";
/// Journal segment id.
pub const SEGMENT: &str = "segment";
/// Activation fence token.
pub const FENCE: &str = "fence";
/// Wire frame or enum variant name.
pub const FRAME: &str = "frame";
/// Wire/transport family.
pub const WIRE: &str = "wire";
/// Domain operation name.
pub const OPERATION: &str = "operation";
/// Lifecycle or recovery step.
pub const STEP: &str = "step";
/// Operation outcome.
pub const OUTCOME: &str = "outcome";

/// Common span names used by substrate/host boundaries.
pub mod span {
    pub const API_HTTP_REQUEST: &str = "api.http.request";
    pub const API_UNIX_REQUEST: &str = "api.unix.request";
    pub const CUT_RECV: &str = "cut.recv";
    pub const CUT_RUN_TURN: &str = "cut.run_turn";
    pub const CUT_STORE_BROKER: &str = "cut.store.broker";
    pub const CUT_CRED_BROKER: &str = "cut.cred.broker";
    pub const TRANSPORT_REQUEST: &str = "transport.request";
    pub const TRANSPORT_REPLY: &str = "transport.reply";
}

/// Common event names used by substrate/host boundaries.
pub mod event {
    pub const API_REQUEST: &str = "api.request";
    pub const CUT_FRAME_IN: &str = "cut.frame.in";
    pub const CUT_FRAME_OUT: &str = "cut.frame.out";
    pub const TRANSPORT_REQUEST: &str = "transport.request";
    pub const TRANSPORT_REPLY: &str = "transport.reply";
}

/// [OpenTelemetry GenAI semantic-convention](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)
/// attribute names, recorded on the engine's `tracing` spans and mapped 1:1 to OTel span attributes
/// by `tracing-opentelemetry` when the `otel` export layer is installed. Keeping the (dotted) spec
/// spelling here means every call site records the exact registry key.
///
/// Content-bearing attributes (`INPUT_MESSAGES`, `OUTPUT_MESSAGES`, `SYSTEM_INSTRUCTIONS`,
/// `AGENT_DESCRIPTION`, tool arguments/results) may carry user/PII data; they are only ever recorded
/// when the export layer is compiled in.
pub mod gen_ai {
    /// The operation name (`chat`, `execute_tool`, `invoke_agent`).
    pub const OPERATION_NAME: &str = "gen_ai.operation.name";
    /// The conversation/session id used to correlate turns.
    pub const CONVERSATION_ID: &str = "gen_ai.conversation.id";
    /// The agent's unique id (the daemon profile id).
    pub const AGENT_ID: &str = "gen_ai.agent.id";
    /// Free-form description of the agent — the daemon persona ("SOUL"): the profile's system prompt.
    pub const AGENT_DESCRIPTION: &str = "gen_ai.agent.description";
    /// The system instructions actually sent to the model this call (persona + composed context).
    pub const SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";
    /// The chat history provided to the model as input (GenAI message-schema JSON).
    pub const INPUT_MESSAGES: &str = "gen_ai.input.messages";
    /// The messages the model returned (GenAI message-schema JSON).
    pub const OUTPUT_MESSAGES: &str = "gen_ai.output.messages";

    /// The requested model id.
    pub const REQUEST_MODEL: &str = "gen_ai.request.model";
    /// Whether the request was made in streaming mode.
    pub const REQUEST_STREAM: &str = "gen_ai.request.stream";
    /// Sampling temperature.
    pub const REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
    /// Nucleus (top-p) sampling cutoff.
    pub const REQUEST_TOP_P: &str = "gen_ai.request.top_p";
    /// Top-k sampling cutoff.
    pub const REQUEST_TOP_K: &str = "gen_ai.request.top_k";
    /// The maximum number of output tokens requested.
    pub const REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
    /// The sampling seed, when set.
    pub const REQUEST_SEED: &str = "gen_ai.request.seed";

    /// The GenAI provider vendor (`openai`, `anthropic`, `gcp.gemini`, …).
    pub const PROVIDER_NAME: &str = "gen_ai.provider.name";
    /// The provider-reported response/completion id.
    pub const RESPONSE_ID: &str = "gen_ai.response.id";
    /// The provider-reported response model id.
    pub const RESPONSE_MODEL: &str = "gen_ai.response.model";
    /// The model's stop reason(s) (`stop`, `length`, `tool_calls`, …).
    pub const RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
    /// Time to the first streamed chunk, in seconds.
    pub const RESPONSE_TIME_TO_FIRST_CHUNK: &str = "gen_ai.response.time_to_first_chunk";

    /// Input/prompt tokens (includes cached).
    pub const USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    /// Output/completion tokens.
    pub const USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    /// Input tokens served from a provider-managed cache.
    pub const USAGE_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";
    /// Input tokens written to a provider-managed cache.
    pub const USAGE_CACHE_CREATION_INPUT_TOKENS: &str = "gen_ai.usage.cache_creation.input_tokens";
    /// Output tokens spent on reasoning.
    pub const USAGE_REASONING_OUTPUT_TOKENS: &str = "gen_ai.usage.reasoning.output_tokens";

    /// The tool name utilized by the agent.
    pub const TOOL_NAME: &str = "gen_ai.tool.name";
    /// The tool type (`function`).
    pub const TOOL_TYPE: &str = "gen_ai.tool.type";
    /// The tool call identifier.
    pub const TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
    /// The tool call arguments (may contain sensitive data).
    pub const TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
    /// The tool call result (may contain sensitive data).
    pub const TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";
    /// The list of tool definitions available to the model (GenAI tool-definition JSON).
    pub const TOOL_DEFINITIONS: &str = "gen_ai.tool.definitions";
}

/// Daemon-specific attributes that have no standard `gen_ai.*` home (cost/call accounting). Kept in
/// a `daemon.*` namespace so they never collide with a future OpenTelemetry attribute.
pub mod daemon {
    /// Estimated cost of a model call/turn in micro-USD (millionths of a dollar).
    pub const USAGE_COST_MICROS: &str = "daemon.usage.cost_micros";
    /// Provider API calls made by a model call/turn.
    pub const USAGE_API_CALLS: &str = "daemon.usage.api_calls";

    /// User-feedback attribute names for the `app.feedback` OTel log event (see
    /// [`crate::feedback`]). Feedback-specific fields have no OpenTelemetry semantic-convention
    /// home, so they live under `daemon.feedback.*`; the fields that DO map to a convention
    /// (session id, turn model/provider/finish-reason/token usage) reuse the `gen_ai.*` +
    /// [`TRACE_ID`](super::TRACE_ID) names instead of duplicating them here.
    pub mod feedback {
        /// The feedback category: `response` (on a specific agent turn) or `app` (general).
        pub const KIND: &str = "daemon.feedback.kind";
        /// The thumbs rating, when given: `up` or `down`.
        pub const RATING: &str = "daemon.feedback.rating";
        /// The free-form comment body, when supplied (may carry user-authored text).
        pub const COMMENT: &str = "daemon.feedback.comment";
        /// The UI surface the feedback was submitted from (e.g. a page/panel identifier).
        pub const SURFACE: &str = "daemon.feedback.surface";
        /// The consent posture under which this event is exported: `opted-in` (telemetry enabled,
        /// a long-lived provider is reused) or `explicit-one-shot` (telemetry otherwise disabled;
        /// the user explicitly submitted, so a scoped provider is built, flushed, and shut down).
        pub const CONSENT: &str = "daemon.feedback.consent";
        /// The submitting client/app version, when known.
        pub const APP_VERSION: &str = "daemon.feedback.app_version";
        /// The submitting client operating system, when known.
        pub const OS: &str = "daemon.feedback.os";
        /// The node (`daemon-node`) version that exported the event.
        pub const NODE_VERSION: &str = "daemon.feedback.node_version";
        /// When the feedback was created, in Unix epoch milliseconds.
        pub const CREATED_AT_MS: &str = "daemon.feedback.created_at_ms";
        /// The rated response text, carried only when the submitter consented via `include_content`
        /// (per-event consent). Makes a response thumb self-describing rather than a bare anchor.
        pub const CONTENT: &str = "daemon.feedback.content";
    }
}
