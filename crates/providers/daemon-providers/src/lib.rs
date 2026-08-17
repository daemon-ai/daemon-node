// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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
mod local;

pub use genai_provider::{
    discovery_vendor_ids, genai_listed_models, genai_models_for, genai_models_for_id,
    genai_models_for_id_classified, genai_models_for_result, project_auth,
    vendor_cloud_credentials, vendor_keyless, GenAiEmbedder, GenAiProvider, VendorListing,
    DAEMON_CLOUD_BASE, DISCOVERY_ADAPTERS,
};
pub use local::{
    probe_worker_engines, EngineProbe, LocalEmbedder, LocalInferenceState, LocalInferenceStatus,
    LocalProvider, SwitchableLocalProvider, ToolAdvertisement, WorkerConfig,
};

use daemon_common::UsageDelta;
use daemon_core::{
    classify_api_error, repair_tool_args, repair_tool_call, scrub_content, Failure, ModelOutput,
    ToolCall,
};

/// The conservative output-token cap applied when a model declares no published maximum (E5). Both
/// provider families source their per-generation output cap from model metadata when known and fall
/// back to this only for unknown models — so a large-output model is never silently clamped, while an
/// unknown one still gets a sane bound. Cloud overrides come from [`genai_provider::known_max_output`];
/// local overrides come from the configured worker cap (else this, bounded by the context window).
pub(crate) const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

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
        ..Default::default()
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

/// The empty-assembly gate (session-unification §9): a request with zero messages is OUR bug (a
/// turn ran on an un-seeded conversation), and every cloud API deterministically 400s an empty
/// `messages` array — which the recovery loop then retried during the incident. Refuse locally,
/// before the wire, as the same non-retryable [`Failure::InvalidRequest`] a provider 400 maps to.
/// Deliberately at the networked-provider boundary, NOT in the engine: scripted/mock providers
/// (the conformance harness's orchestrators) legitimately drive blank-session turns.
pub(crate) fn empty_assembly_gate(req: &daemon_core::Request) -> Result<(), Failure> {
    if req.messages.is_empty() {
        return Err(Failure::InvalidRequest(
            "empty request assembly: refusing the provider call (a turn ran on a conversation \
             with no messages — a seeding/creation bug upstream)"
                .into(),
        ));
    }
    Ok(())
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
        E::HttpError { status, body, .. } => classify_api_error(status.as_u16(), |_| None, &body),
        E::StreamParse { .. } | E::InvalidJsonResponseElement { .. } => {
            Failure::FormatError(format!("genai decode: {err}"))
        }
        // A streaming-path failure. genai's stream layer checks the HTTP status BEFORE streaming
        // and, on a non-2xx, boxes a `genai::Error::HttpError { status, body, .. }` into
        // `WebStream.error` — so an HTTP reject on the streaming path (the incident's 400) carries
        // its status here and MUST route through `classify_api_error` like every other HTTP error
        // (session-unification §9), not blanket-retry as transport. Only a genuinely mid-stream
        // transport failure (reset, hung stream) stays `TransientTransport`.
        E::WebStream { cause, error, .. } => match error.downcast::<genai::Error>() {
            Ok(inner) => match *inner {
                E::HttpError { status, body, .. } => {
                    classify_api_error(status.as_u16(), |_| None, &body)
                }
                other => Failure::TransientTransport(format!("genai stream: {other}")),
            },
            Err(_) => Failure::TransientTransport(format!("genai stream: {cause}")),
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    type BoxErr = Box<dyn std::error::Error + Send + Sync>;

    fn model_iden() -> genai::ModelIden {
        genai::ModelIden::new(genai::adapter::AdapterKind::OpenAI, "test-model")
    }

    fn web_stream(error: BoxErr) -> genai::Error {
        genai::Error::WebStream {
            model_iden: model_iden(),
            cause: "stream open failed".into(),
            error,
        }
    }

    /// §9 regression (the incident's masked classification): an HTTP reject on the *streaming*
    /// path boxes a `genai::Error::HttpError` inside `WebStream` — its status MUST route through
    /// `classify_api_error` exactly like the non-streaming path, not blanket-retry as transport.
    #[test]
    fn web_stream_http_reject_routes_through_status_classification() {
        let http_400 = genai::Error::HttpError {
            status: reqwest::StatusCode::BAD_REQUEST,
            canonical_reason: "Bad Request".into(),
            body: r#"{"error":{"message":"messages: at least one message is required"}}"#.into(),
        };
        assert!(
            matches!(
                classify_genai_error(web_stream(Box::new(http_400))),
                Failure::InvalidRequest(_)
            ),
            "a streamed 400 must abort as InvalidRequest, not retry as transport"
        );

        let http_429 = genai::Error::HttpError {
            status: reqwest::StatusCode::TOO_MANY_REQUESTS,
            canonical_reason: "Too Many Requests".into(),
            body: "rate limited".into(),
        };
        assert!(matches!(
            classify_genai_error(web_stream(Box::new(http_429))),
            Failure::RateLimit { .. }
        ));
    }

    /// A genuinely mid-stream transport failure (no boxed HTTP status) stays retryable transport.
    #[test]
    fn web_stream_transport_failure_stays_transient() {
        let io: BoxErr = Box::new(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "reset",
        ));
        assert!(matches!(
            classify_genai_error(web_stream(io)),
            Failure::TransientTransport(_)
        ));
    }

    /// §9: zero assembled messages never reaches the wire — refused locally as the same
    /// non-retryable failure a provider 400 maps to.
    #[test]
    fn empty_assembly_refused_before_the_wire() {
        let empty = daemon_core::Request::default();
        assert!(matches!(
            empty_assembly_gate(&empty),
            Err(Failure::InvalidRequest(_))
        ));

        let seeded = daemon_core::Request {
            messages: vec![daemon_core::RequestMsg {
                role: "user".into(),
                content: "hello".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(empty_assembly_gate(&seeded).is_ok());
    }
}
