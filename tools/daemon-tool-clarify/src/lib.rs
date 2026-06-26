//! `daemon-tool-clarify` — the `clarify` chat tool (`daemon_core::Tool`): a first-class
//! human-in-the-loop ask. When the agent needs a decision only the user can make, it asks a
//! question (optionally with fixed options) and **blocks** on the §17 host request channel until the
//! user answers, then returns that answer into the turn.
//!
//! A question with `options` raises a [`HostRequestKind::Choice`] (the user picks one); a question
//! without options raises a [`HostRequestKind::Input`] (free-form). The answer originates from the
//! user (the session principal), so it is returned as ordinary trusted content — not untrusted
//! external data.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_common::ReqId;
use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
use daemon_protocol::{HostRequest, HostRequestKind, HostResponseBody, ToolDetail};
use serde::{Deserialize, Serialize};

/// The JSON-Schema advertised for the `clarify` tool.
const CLARIFY_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["question"],
  "properties": {
    "question": {"type": "string", "description": "The question to ask the user."},
    "options": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Optional fixed choices. If present, the user picks one; otherwise the answer is free-form."
    }
  }
}"#;

/// The `clarify` tool.
#[derive(Default)]
pub struct ClarifyTool;

impl ClarifyTool {
    /// A clarify tool.
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    question: String,
    #[serde(default)]
    options: Option<Vec<String>>,
}

/// The structured payload attached for the GUI to render the clarify exchange.
#[derive(Serialize)]
struct ClarifyDetail<'a> {
    question: &'a str,
    options: &'a [String],
    answer: &'a str,
}

#[async_trait]
impl Tool for ClarifyTool {
    fn name(&self) -> &str {
        "clarify"
    }

    fn schema(&self) -> &str {
        CLARIFY_SCHEMA
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let args: Args = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("clarify: invalid arguments: {e}"),
                )
            }
        };
        if args.question.trim().is_empty() {
            return ToolOutcome::text(call.call_id.clone(), false, "clarify: empty question");
        }
        let options = args.options.unwrap_or_default();
        let non_empty_options: Vec<String> = options
            .iter()
            .filter(|o| !o.trim().is_empty())
            .cloned()
            .collect();

        let kind = if non_empty_options.is_empty() {
            HostRequestKind::Input {
                prompt: args.question.clone(),
            }
        } else {
            HostRequestKind::Choice {
                prompt: args.question.clone(),
                options: non_empty_options.clone(),
            }
        };
        // Block on the host until the user answers (§17). The host assigns the real request id.
        let resp = cx
            .host
            .request(HostRequest {
                request_id: ReqId(0),
                kind,
            })
            .await;

        let answer = match resp.body {
            HostResponseBody::Input(text) => text,
            HostResponseBody::Chosen(idx) => non_empty_options
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("option {idx}")),
            // Any other reply (e.g. a declined approval) means the user did not answer.
            _ => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    "clarify: no answer provided",
                )
            }
        };

        let detail = ToolDetail {
            kind: "clarify".to_string(),
            body: serde_json::to_vec(&ClarifyDetail {
                question: &args.question,
                options: &non_empty_options,
                answer: &answer,
            })
            .unwrap_or_default(),
        };
        ToolOutcome::text(
            call.call_id.clone(),
            true,
            format!("{}\n\nuser answer: {answer}", args.question),
        )
        .with_detail(detail)
    }
}
