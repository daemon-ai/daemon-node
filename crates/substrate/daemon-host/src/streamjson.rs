// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`StreamJsonCodec`] — the Claude-Code `stream-json` foreign codec (the second [`Codec`]).
//!
//! Claude Code (and compatible agents — Amp, Cursor's CLI) emit newline-delimited JSON on stdout
//! when run with `--output-format stream-json`, and read NDJSON user turns on stdin with
//! `--input-format stream-json`. The envelope is a handful of `{"type": ...}` records carrying
//! Anthropic content blocks; this codec translates that one-way event stream into §17
//! [`Outbound`] frames and our §17 [`Inbound`] commands back into the input lines the agent expects.
//!
//! It is **stateless about the protocol but stateful about the turn**: it assigns the monotonic
//! event `seq` the §17 surface requires (stream-json carries none), synthesizes a `TurnStarted` at
//! the first content of a turn, and closes the turn on `result`. A blocking permission prompt
//! (`control_request`) becomes an [`Outbound::Request`]; the host's reply is encoded back as a
//! `control_response`. Forward-compatible by construction: unknown record `type`s and unknown fields
//! are ignored (per the vendors' documented contract), so a newer agent never breaks the bridge.

use crate::foreign::Codec;
use daemon_common::{ReqId, UsageDelta};
use daemon_protocol::{
    AgentCommand, AgentEvent, EndReason, HostRequest, HostRequestKind, HostResponseBody, Inbound,
    Outbound, ToolCallView, ToolDetail, ToolResultView, TurnSummary, TurnTrigger,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

/// The Claude-Code `stream-json` codec. One per session (it carries the turn's `seq` cursor and the
/// permission-request correlation map).
#[derive(Default)]
pub struct StreamJsonCodec {
    /// The monotonic §17 event sequence counter (stream-json events carry none).
    seq: u64,
    /// Whether a turn is currently open (a `TurnStarted` has been synthesized, no `result` yet).
    turn_open: bool,
    /// The next synthetic [`ReqId`] for an agent-raised permission request.
    next_req: u64,
    /// Maps a synthetic [`ReqId`] back to the agent's original string `request_id`, so the host's
    /// reply can be addressed to the right `control_request`.
    pending: HashMap<u64, String>,
}

impl StreamJsonCodec {
    /// Construct a fresh codec.
    pub fn new() -> Self {
        Self::default()
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    /// Synthesize a `TurnStarted` if no turn is open (stream-json has no explicit turn-start record).
    fn ensure_turn_started(&mut self, out: &mut Vec<Outbound>) {
        if !self.turn_open {
            self.turn_open = true;
            let seq = self.next_seq();
            out.push(Outbound::Event(AgentEvent::TurnStarted {
                seq,
                trigger: TurnTrigger::User,
            }));
        }
    }

    fn decode_content(&mut self, content: Content, out: &mut Vec<Outbound>) {
        match content {
            Content::Empty => {}
            Content::Text(text) => {
                let seq = self.next_seq();
                out.push(Outbound::Event(AgentEvent::TextDelta { seq, text }));
            }
            Content::Blocks(blocks) => {
                for block in blocks {
                    self.decode_block(block, out);
                }
            }
        }
    }

    fn decode_block(&mut self, block: Block, out: &mut Vec<Outbound>) {
        match block.ty.as_str() {
            "text" => {
                if let Some(text) = block.text {
                    let seq = self.next_seq();
                    out.push(Outbound::Event(AgentEvent::TextDelta { seq, text }));
                }
            }
            "thinking" => {
                if let Some(text) = block.thinking {
                    let seq = self.next_seq();
                    out.push(Outbound::Event(AgentEvent::ReasoningDelta { seq, text }));
                }
            }
            "tool_use" => {
                let name = block.name.unwrap_or_default();
                let call_id = block.id.unwrap_or_default();
                let args_summary = block.input.as_ref().map(summarize).unwrap_or_default();
                let detail = block
                    .input
                    .as_ref()
                    .map(|v| ToolDetail::new(name.clone(), cbor_bytes(v)));
                let seq = self.next_seq();
                out.push(Outbound::Event(AgentEvent::ToolStarted {
                    seq,
                    call: ToolCallView {
                        call_id,
                        name,
                        args_summary,
                        detail,
                    },
                }));
            }
            "tool_result" => {
                let call_id = block.tool_use_id.unwrap_or_default();
                let ok = !block.is_error.unwrap_or(false);
                let summary = block.content.as_ref().map(summarize).unwrap_or_default();
                let detail = block
                    .content
                    .as_ref()
                    .map(|v| ToolDetail::new("tool_result", cbor_bytes(v)));
                let seq = self.next_seq();
                out.push(Outbound::Event(AgentEvent::ToolFinished {
                    seq,
                    result: ToolResultView {
                        call_id,
                        ok,
                        summary,
                        detail,
                    },
                }));
            }
            // Forward-compatible: ignore unknown content block types.
            _ => {}
        }
    }
}

impl Codec for StreamJsonCodec {
    fn decode(&mut self, msg: &[u8]) -> Vec<Outbound> {
        // A blank keep-alive line or an unparseable record is ignored (forward-compatible).
        let env: Envelope = match serde_json::from_slice(msg) {
            Ok(env) => env,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        match env.ty.as_str() {
            // `system`/`init` carries session setup (cwd, tools, model) — no §17 projection.
            "system" => {}
            "assistant" | "user" => {
                self.ensure_turn_started(&mut out);
                if let Some(message) = env.message {
                    self.decode_content(message.content, &mut out);
                }
            }
            "result" => {
                self.ensure_turn_started(&mut out);
                let end_reason = if env.is_error.unwrap_or(false) {
                    EndReason::Failed
                } else {
                    EndReason::Completed
                };
                let seq = self.next_seq();
                self.turn_open = false;
                out.push(Outbound::Event(AgentEvent::TurnFinished {
                    seq,
                    summary: TurnSummary {
                        end_reason,
                        final_text: env.result,
                        usage: UsageDelta::default(),
                    },
                }));
            }
            "control_request" => {
                if let (Some(orig), Some(request)) = (env.request_id, env.request) {
                    let local = self.next_req;
                    self.next_req += 1;
                    self.pending.insert(local, orig);
                    let prompt = format!(
                        "{}: {}",
                        request.subtype.as_deref().unwrap_or("permission"),
                        request.tool_name.as_deref().unwrap_or("(tool)"),
                    );
                    out.push(Outbound::Request(HostRequest {
                        request_id: ReqId(local),
                        kind: HostRequestKind::Approval { prompt },
                    }));
                }
            }
            // Forward-compatible: ignore unknown record types.
            _ => {}
        }
        out
    }

    fn encode(&mut self, inbound: Inbound) -> Vec<Vec<u8>> {
        match inbound {
            Inbound::Command(AgentCommand::StartTurn { input, .. }) => vec![user_line(&input.text)],
            Inbound::Command(AgentCommand::Steer { text, .. }) => vec![user_line(&text)],
            Inbound::Response(resp) => match self.pending.remove(&resp.request_id.0) {
                Some(orig) => {
                    let allow = matches!(resp.body, HostResponseBody::Approved(true));
                    vec![control_response_line(&orig, allow)]
                }
                // A reply with no recorded `control_request` (e.g. a non-permission response) has no
                // stream-json input form.
                None => Vec::new(),
            },
            // `Interrupt`/`Snapshot`/`Shutdown` have no in-band stream-json input form; the child is
            // killed on unit drop (kill-on-drop `ChildGuard`).
            _ => Vec::new(),
        }
    }
}

/// Encode a user turn line in the `stream-json` input dialect.
fn user_line(text: &str) -> Vec<u8> {
    let value = json!({
        "type": "user",
        "message": { "role": "user", "content": text },
    });
    serde_json::to_vec(&value).expect("serialize stream-json user line")
}

/// Encode a permission decision as a `control_response` line.
fn control_response_line(request_id: &str, allow: bool) -> Vec<u8> {
    let value = json!({
        "type": "control_response",
        "request_id": request_id,
        "response": {
            "subtype": "success",
            "behavior": if allow { "allow" } else { "deny" },
        },
    });
    serde_json::to_vec(&value).expect("serialize stream-json control_response line")
}

/// CBOR-encode an opaque JSON payload for a [`ToolDetail`] body (CBOR by convention).
fn cbor_bytes(value: &serde_json::Value) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).expect("cbor-encode tool detail");
    buf
}

/// A short, non-secret human summary of a JSON payload for the coarse management view.
fn summarize(value: &serde_json::Value) -> String {
    let mut s = value.to_string();
    const MAX: usize = 200;
    if s.len() > MAX {
        s.truncate(MAX);
        s.push('…');
    }
    s
}

// ---------------------------------------------------------------------------
// Wire schema (lenient: unknown fields ignored, no `deny_unknown_fields`)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Envelope {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    message: Option<AnthropicMessage>,
    #[serde(default)]
    is_error: Option<bool>,
    #[serde(default)]
    result: Option<String>,
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    request: Option<ControlRequest>,
}

#[derive(Deserialize)]
struct AnthropicMessage {
    #[serde(default)]
    content: Content,
}

/// Anthropic message content: a bare string or an array of typed blocks.
#[derive(Deserialize, Default)]
#[serde(untagged)]
enum Content {
    #[default]
    Empty,
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Deserialize)]
struct Block {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
    #[serde(default)]
    tool_use_id: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    is_error: Option<bool>,
}

#[derive(Deserialize)]
struct ControlRequest {
    #[serde(default)]
    subtype: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_protocol::HostResponse;

    #[test]
    fn assistant_then_result_maps_to_turn_lifecycle() {
        let mut codec = StreamJsonCodec::new();
        assert!(codec
            .decode(br#"{"type":"system","subtype":"init","model":"x"}"#)
            .is_empty());

        let assistant = codec.decode(
            br#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}"#,
        );
        // A synthesized TurnStarted then the text delta.
        assert!(matches!(
            assistant[0],
            Outbound::Event(AgentEvent::TurnStarted { .. })
        ));
        assert!(matches!(
            &assistant[1],
            Outbound::Event(AgentEvent::TextDelta { text, .. }) if text == "hi"
        ));

        let result = codec
            .decode(br#"{"type":"result","subtype":"success","is_error":false,"result":"done"}"#);
        assert!(matches!(
            &result[0],
            Outbound::Event(AgentEvent::TurnFinished { summary, .. })
                if summary.end_reason == EndReason::Completed && summary.final_text.as_deref() == Some("done")
        ));
    }

    #[test]
    fn tool_use_carries_opaque_detail() {
        let mut codec = StreamJsonCodec::new();
        let out = codec.decode(
            br#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"cmd":"ls"}}]}}"#,
        );
        let started = out
            .iter()
            .find_map(|o| match o {
                Outbound::Event(AgentEvent::ToolStarted { call, .. }) => Some(call),
                _ => None,
            })
            .expect("a tool_use block maps to ToolStarted");
        assert_eq!(started.name, "Bash");
        assert_eq!(started.call_id, "toolu_1");
        let detail = started.detail.as_ref().expect("opaque input detail");
        assert_eq!(detail.kind, "Bash");
        assert!(!detail.body.is_empty());
    }

    #[test]
    fn permission_request_round_trips_to_control_response() {
        let mut codec = StreamJsonCodec::new();
        let out = codec.decode(
            br#"{"type":"control_request","request_id":"req_7","request":{"subtype":"can_use_tool","tool_name":"Bash"}}"#,
        );
        let req = match &out[0] {
            Outbound::Request(req) => req.clone(),
            other => panic!("expected a HostRequest, got {other:?}"),
        };
        assert!(matches!(req.kind, HostRequestKind::Approval { .. }));

        let lines = codec.encode(Inbound::Response(HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved(true),
        }));
        let line = String::from_utf8(lines[0].clone()).unwrap();
        assert!(line.contains("control_response"));
        assert!(line.contains("req_7"));
        assert!(line.contains("allow"));
    }

    #[test]
    fn start_turn_encodes_a_user_line() {
        let mut codec = StreamJsonCodec::new();
        let lines = codec.encode(Inbound::Command(AgentCommand::StartTurn {
            input: daemon_protocol::UserMsg::new("do it"),
            request_id: ReqId(0),
        }));
        let line = String::from_utf8(lines[0].clone()).unwrap();
        assert!(line.contains("\"type\":\"user\""));
        assert!(line.contains("do it"));
    }

    #[test]
    fn unknown_record_type_is_ignored() {
        let mut codec = StreamJsonCodec::new();
        assert!(codec
            .decode(br#"{"type":"some_future_thing","foo":1}"#)
            .is_empty());
    }
}
