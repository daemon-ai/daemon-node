// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! OpenTelemetry GenAI span enrichment (feature `otel`).
//!
//! Records [OpenTelemetry GenAI semantic-convention](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)
//! attributes — including full prompt/completion/tool content — onto the engine's turn/model/tool
//! `tracing` spans, from where `tracing-opentelemetry` maps them to OTel span attributes.
//!
//! Two gates keep this inert unless telemetry export is actually running:
//! - **Compile-time:** the whole module (and every call site) is behind the `otel` feature.
//! - **Runtime:** [`set_genai_capture`] must be turned on by the host once an OTLP exporter is
//!   installed. Until then [`capture_enabled`] is `false` and every recorder is a no-op, so a build
//!   with `--features otel` but no configured endpoint records nothing (no content on any span, no
//!   change to `fmt` logs).
//!
//! Content-bearing attributes carry raw user/model text; they are only ever attached when both gates
//! are open.

use crate::conversation::ToolCall;
use crate::provider::{ModelOutput, Request, RequestMsg};
use crate::tools::ToolDef;
use daemon_protocol::{EndReason, TurnSummary};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::Span;

/// Runtime capture switch. Off until the host installs an OTLP exporter and calls
/// [`set_genai_capture(true)`](set_genai_capture).
static GENAI_CAPTURE: AtomicBool = AtomicBool::new(false);

/// Turn GenAI span enrichment on/off at runtime. The host calls this with `true` after installing an
/// OpenTelemetry exporter (see `daemon_telemetry::init_telemetry`), and it stays off otherwise.
pub fn set_genai_capture(on: bool) {
    GENAI_CAPTURE.store(on, Ordering::Relaxed);
}

/// Whether GenAI span enrichment is active (compiled in **and** switched on).
pub fn capture_enabled() -> bool {
    GENAI_CAPTURE.load(Ordering::Relaxed)
}

/// Whether recording onto `span` would be observed (capture on and the span is enabled by some
/// layer's filter). Recorders bail early on `false` so no content is serialized needlessly.
fn active(span: &Span) -> bool {
    capture_enabled() && !span.is_disabled()
}

/// Best-effort parse of a tool/argument JSON string into structured JSON; falls back to the raw
/// string so malformed args still export.
fn parse_json(raw: &str) -> Value {
    serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

fn input_message(msg: &RequestMsg) -> Value {
    let mut parts: Vec<Value> = Vec::new();
    if msg.role == "tool" {
        parts.push(json!({
            "type": "tool_call_response",
            "id": msg.tool_call_id.clone().unwrap_or_default(),
            "result": msg.content,
        }));
    } else {
        if !msg.content.is_empty() {
            parts.push(json!({ "type": "text", "content": msg.content }));
        }
        for tc in &msg.tool_calls {
            parts.push(json!({
                "type": "tool_call",
                "id": tc.call_id,
                "name": tc.name,
                "arguments": parse_json(&tc.args),
            }));
        }
    }
    json!({ "role": msg.role, "parts": parts })
}

fn output_message(out: &ModelOutput) -> Value {
    let mut parts: Vec<Value> = Vec::new();
    if !out.text.is_empty() {
        parts.push(json!({ "type": "text", "content": out.text }));
    }
    if let Some(reasoning) = &out.reasoning {
        if !reasoning.is_empty() {
            parts.push(json!({ "type": "reasoning", "content": reasoning }));
        }
    }
    for tc in &out.tool_calls {
        parts.push(json!({
            "type": "tool_call",
            "id": tc.call_id,
            "name": tc.name,
            "arguments": parse_json(&tc.args),
        }));
    }
    let mut msg = json!({ "role": "assistant", "parts": parts });
    if let Some(finish) = out.meta.as_deref().and_then(|m| m.finish_reason.as_ref()) {
        msg["finish_reason"] = json!(finish);
    }
    msg
}

fn stringify(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// Map the turn-level [`EndReason`] to a stable lowercase label (recorded under a `daemon.*` key —
/// this is the turn-loop outcome, distinct from the model's `gen_ai.response.finish_reasons`).
fn end_reason_str(reason: &EndReason) -> &'static str {
    match reason {
        EndReason::Completed => "completed",
        EndReason::Suspended => "suspended",
        EndReason::Interrupted => "interrupted",
        EndReason::BudgetExhausted => "budget_exhausted",
        EndReason::NoProgress => "no_progress",
        EndReason::Failed => "failed",
        // `EndReason` is `#[non_exhaustive]`; label any future variant generically.
        _ => "other",
    }
}

/// Record the session/agent identity onto the turn (root) span: operation, conversation id, agent id
/// + description (the persona / "SOUL"), and the requested model.
pub(crate) fn record_turn_identity(
    span: &Span,
    conversation_id: &str,
    agent_id: &str,
    model: &str,
    persona: &str,
) {
    if !active(span) {
        return;
    }
    span.record("gen_ai.operation.name", "invoke_agent");
    span.record("gen_ai.conversation.id", conversation_id);
    span.record("gen_ai.agent.id", agent_id);
    span.record("gen_ai.request.model", model);
    if !persona.is_empty() {
        span.record("gen_ai.agent.description", persona);
    }
}

/// Record the folded turn usage + outcome onto the turn (root) span.
pub(crate) fn record_turn_summary(span: &Span, summary: &TurnSummary) {
    if !active(span) {
        return;
    }
    let usage = &summary.usage;
    span.record("gen_ai.usage.input_tokens", usage.input_tokens);
    span.record("gen_ai.usage.output_tokens", usage.output_tokens);
    span.record(
        "gen_ai.usage.cache_read.input_tokens",
        usage.cache_read_tokens,
    );
    span.record(
        "gen_ai.usage.cache_creation.input_tokens",
        usage.cache_write_tokens,
    );
    span.record(
        "gen_ai.usage.reasoning.output_tokens",
        usage.reasoning_tokens,
    );
    span.record("daemon.usage.cost_micros", usage.cost_micros);
    span.record("daemon.usage.api_calls", u64::from(usage.api_calls));
    span.record(
        "daemon.turn.end_reason",
        end_reason_str(&summary.end_reason),
    );
}

/// Record the model request (input messages, system instructions, offered tool definitions) onto a
/// model-call span, before the provider consumes the request.
pub(crate) fn record_model_request(span: &Span, req: &Request) {
    if !active(span) {
        return;
    }
    span.record("gen_ai.operation.name", "chat");
    span.record("gen_ai.request.stream", true);
    let system = json!([{ "type": "text", "content": req.system }]);
    span.record("gen_ai.system_instructions", stringify(&system).as_str());
    let messages: Vec<Value> = req.messages.iter().map(input_message).collect();
    span.record(
        "gen_ai.input.messages",
        stringify(&Value::Array(messages)).as_str(),
    );
    if !req.tools.is_empty() {
        let defs: Vec<Value> = req.tools.iter().map(tool_definition).collect();
        span.record(
            "gen_ai.tool.definitions",
            stringify(&Value::Array(defs)).as_str(),
        );
    }
}

fn tool_definition(def: &ToolDef) -> Value {
    json!({
        "type": "function",
        "name": def.name,
        "parameters": parse_json(&def.schema),
    })
}

/// Record the model response (output messages, token usage, response metadata, effective sampling
/// params) onto a model-call span.
pub(crate) fn record_model_response(span: &Span, out: &ModelOutput) {
    if !active(span) {
        return;
    }
    span.record(
        "gen_ai.output.messages",
        stringify(&Value::Array(vec![output_message(out)])).as_str(),
    );
    let usage = &out.usage;
    span.record("gen_ai.usage.input_tokens", usage.input_tokens);
    span.record("gen_ai.usage.output_tokens", usage.output_tokens);
    span.record(
        "gen_ai.usage.cache_read.input_tokens",
        usage.cache_read_tokens,
    );
    span.record(
        "gen_ai.usage.cache_creation.input_tokens",
        usage.cache_write_tokens,
    );
    span.record(
        "gen_ai.usage.reasoning.output_tokens",
        usage.reasoning_tokens,
    );
    span.record("daemon.usage.cost_micros", usage.cost_micros);
    span.record("daemon.usage.api_calls", u64::from(usage.api_calls));
    let Some(meta) = out.meta.as_deref() else {
        return;
    };
    if let Some(reason) = &meta.finish_reason {
        span.record("gen_ai.response.finish_reasons", reason.as_str());
    }
    if let Some(id) = &meta.response_id {
        span.record("gen_ai.response.id", id.as_str());
    }
    if let Some(model) = &meta.response_model {
        span.record("gen_ai.response.model", model.as_str());
    }
    if let Some(provider) = &meta.provider_name {
        span.record("gen_ai.provider.name", provider.as_str());
    }
    if let Some(params) = &meta.params {
        if let Some(t) = params.temperature {
            span.record("gen_ai.request.temperature", t);
        }
        if let Some(p) = params.top_p {
            span.record("gen_ai.request.top_p", p);
        }
        if let Some(k) = params.top_k {
            span.record("gen_ai.request.top_k", u64::from(k));
        }
        if let Some(m) = params.max_tokens {
            span.record("gen_ai.request.max_tokens", u64::from(m));
        }
        if let Some(s) = params.seed {
            span.record("gen_ai.request.seed", s);
        }
    }
}

/// Record the time-to-first-chunk (seconds) onto a model-call span.
pub(crate) fn record_time_to_first_chunk(span: &Span, seconds: f64) {
    if !active(span) {
        return;
    }
    span.record("gen_ai.response.time_to_first_chunk", seconds);
}

/// Record the tool invocation identity + full arguments onto a tool span.
pub(crate) fn record_tool_call(span: &Span, call: &ToolCall) {
    if !active(span) {
        return;
    }
    span.record("gen_ai.operation.name", "execute_tool");
    span.record("gen_ai.tool.type", "function");
    span.record("gen_ai.tool.name", call.name.as_str());
    span.record("gen_ai.tool.call.id", call.call_id.as_str());
    span.record(
        "gen_ai.tool.call.arguments",
        stringify(&parse_json(&call.args)).as_str(),
    );
}

/// Record the tool result content onto a tool span.
pub(crate) fn record_tool_result(span: &Span, content: &str) {
    if !active(span) {
        return;
    }
    span.record(
        "gen_ai.tool.call.result",
        stringify(&parse_json(content)).as_str(),
    );
}
