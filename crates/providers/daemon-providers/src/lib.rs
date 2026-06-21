//! Real networked model providers (§7) — a thin [`Provider`] over the [`genai`] multi-provider
//! client.
//!
//! Rather than hand-roll per-provider HTTP/SSE/JSON, this crate adapts one native-protocol client
//! ([`genai`], covering OpenAI/Anthropic/Gemini/Groq/… with streaming + native tools) **underneath**
//! `daemon-core`'s [`Provider`] trait. Everything that makes the engine robust stays ours:
//!
//! - the §8 [`Failure`] taxonomy + recovery: [`classify_genai_error`] turns a `genai` transport error
//!   into the precise [`Failure`] (reading HTTP status/headers/body via
//!   [`classify_api_error`](daemon_core::classify_api_error)), so [`ModelCallPolicy`] is unchanged;
//! - the §9 repair pipeline: [`finalize_output`] runs at the decode boundary (tool name/arg repair +
//!   think-scrub) on `genai`'s decoded output;
//! - the §10/§11 seams and the streaming contract ([`StreamEvent`]).
//!
//! `genai` owns only the wire: request/response mapping, SSE framing, and reasoning normalization.
//! Native tool **schemas** and **`tool_call_id`** round-trip through the enriched [`Request`].

mod genai_provider;

pub use genai_provider::GenAiProvider;

use daemon_common::UsageDelta;
use daemon_core::{
    classify_api_error, repair_tool_args, repair_tool_call, scrub_content, Failure, ModelOutput,
    ToolCall,
};

/// A tool call as decoded off the wire, before §9 repair.
#[derive(Clone, Debug, Default)]
pub(crate) struct RawToolCall {
    pub id: String,
    pub name: String,
    pub args: String,
}

/// Build the canonical [`ModelOutput`], applying §9 repair: think-scrub the content channel (routing
/// any leaked `<think>` spans to reasoning — usually a no-op since `genai` normalizes reasoning) and
/// repair each tool call's name (fuzzy against the offered tools) and arguments (JSON repair +
/// canonicalize). A name that cannot be resolved is kept as-is so the tool pipeline surfaces a
/// corrective "unknown tool" result the model can fix.
pub(crate) fn finalize_output(
    text: String,
    reasoning: Option<String>,
    raw_calls: Vec<RawToolCall>,
    usage: UsageDelta,
    valid_tools: &[String],
) -> ModelOutput {
    let scrub = scrub_content(&text);
    let mut reasoning_acc = reasoning.unwrap_or_default();
    if !scrub.reasoning.is_empty() {
        if !reasoning_acc.is_empty() {
            reasoning_acc.push('\n');
        }
        reasoning_acc.push_str(&scrub.reasoning);
    }

    let tool_calls = raw_calls
        .into_iter()
        .map(|raw| {
            let call = ToolCall {
                call_id: if raw.id.is_empty() {
                    format!("call-{}", &raw.name)
                } else {
                    raw.id
                },
                name: raw.name,
                args: raw.args,
            };
            match repair_tool_call(call.clone(), valid_tools) {
                Ok(repaired) => repaired,
                // Keep the original name (canonicalizing args) — the pipeline reports unknown-tool.
                Err(_) => ToolCall {
                    call_id: call.call_id,
                    name: call.name,
                    args: repair_tool_args(&call.args).args,
                },
            }
        })
        .collect();

    ModelOutput {
        text: scrub.text,
        reasoning: (!reasoning_acc.is_empty()).then_some(reasoning_acc),
        tool_calls,
        usage,
    }
}

/// A short, single-line snippet of an error body for `Failure` messages (never the whole thing).
fn snippet(body: &str) -> String {
    let one_line: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.len() > 200 {
        format!("{}…", &one_line[..200])
    } else {
        one_line
    }
}

/// Map a [`genai::Error`] into the §8 [`Failure`] taxonomy.
///
/// HTTP errors (the common, recoverable case) carry status/headers/body, which we route through the
/// shared [`classify_api_error`] so recovery behaviour is identical to the hand-rolled providers'.
/// Decode/parse errors become a (retryable) `FormatError`; anything else is an (abort) `Provider`.
pub(crate) fn classify_genai_error(err: genai::Error) -> Failure {
    use genai::Error as E;
    match err {
        E::WebModelCall { webc_error, .. } | E::WebAdapterCall { webc_error, .. } => {
            classify_webc(webc_error)
        }
        E::HttpError { status, body, .. } => {
            classify_api_error(status.as_u16(), |_| None, &body)
        }
        E::StreamParse { .. } | E::InvalidJsonResponseElement { .. } => {
            Failure::FormatError(format!("genai decode: {err}"))
        }
        E::WebStream { .. } => Failure::TransientTransport(format!("genai stream: {err}")),
        E::ChatResponse { body, .. } => Failure::Provider(snippet(&body.to_string())),
        other => Failure::Provider(other.to_string()),
    }
}

/// Map a [`genai::webc::Error`] (the HTTP layer) into the §8 [`Failure`] taxonomy.
fn classify_webc(err: genai::webc::Error) -> Failure {
    use genai::webc::Error as W;
    match err {
        W::ResponseFailedStatus {
            status,
            body,
            headers,
        } => classify_api_error(
            status.as_u16(),
            |name| {
                headers
                    .get(name)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string)
            },
            &body,
        ),
        W::ResponseFailedNotJson { .. } | W::ResponseFailedInvalidJson { .. } => {
            Failure::FormatError(format!("genai body: {err}"))
        }
        W::Reqwest(e) => Failure::TransientTransport(format!("transport: {e}")),
        other => Failure::Provider(other.to_string()),
    }
}
