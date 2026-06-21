//! Turn -> message flattening (`daemon-context-lcm-port-spec.md` §6.2, §14.1).
//!
//! The store's granularity is the *message* (for lossless `store_id` lineage), while the engine's
//! granularity is the *turn*. This module maps a slice of [`Turn`]s to the `messages` rows the store
//! persists, and renders a region of turns into the plain text the summarizer consumes.
//!
//! Milestone note: per the M1-M4 scope, ingest is driven from `compact()` over exactly the region
//! being summarized (so a D0 node references real `store_id`s) rather than per-turn in `before_turn`.
//! Full per-turn transcript ingest (with an `ingest_cursor`) lands with the `lcm_grep`/`lcm_expand`
//! search tools (M6), which need the whole transcript, not just compacted spans.

use crate::store::NewMessage;
use crate::tokens::Tokenizer;
use daemon_core::conversation::{ToolCall, Turn};

/// Flatten conversation turns into `messages` rows (the §6.2 mapping):
/// - `User` -> one `user` row.
/// - `Assistant` -> one `assistant` row (reasoning is dropped — §14.4).
/// - `Tool` -> one `assistant` row (text + `tool_calls` JSON), then one `tool` row per result.
pub fn flatten_turns(turns: &[Turn], tok: &Tokenizer) -> Vec<NewMessage> {
    let mut rows = Vec::new();
    for turn in turns {
        match turn {
            Turn::User(u) => rows.push(NewMessage {
                role: "user".into(),
                content: Some(u.text.clone()),
                token_estimate: tok.count_text(&u.text) as i64,
                ..Default::default()
            }),
            Turn::Assistant(a) => rows.push(NewMessage {
                role: "assistant".into(),
                content: Some(a.text.clone()),
                token_estimate: tok.count_text(&a.text) as i64,
                ..Default::default()
            }),
            Turn::Tool(t) => {
                let calls: Vec<ToolCall> = t.calls.iter().map(|(c, _)| c.clone()).collect();
                let tool_calls_json = serde_json::to_string(&calls).ok();
                rows.push(NewMessage {
                    role: "assistant".into(),
                    content: Some(t.assistant.text.clone()),
                    tool_calls: tool_calls_json,
                    token_estimate: tok.count_text(&t.assistant.text) as i64,
                    ..Default::default()
                });
                for (call, result) in &t.calls {
                    rows.push(NewMessage {
                        role: "tool".into(),
                        content: Some(result.content.clone()),
                        tool_call_id: Some(result.call_id.clone()),
                        tool_name: Some(call.name.clone()),
                        token_estimate: tok.count_text(&result.content) as i64,
                        ..Default::default()
                    });
                }
            }
        }
    }
    rows
}

/// Render a region of turns into the plain text the summarizer consumes (role-tagged lines).
pub fn render_turns(turns: &[Turn]) -> String {
    let mut out = String::new();
    for turn in turns {
        match turn {
            Turn::User(u) => push_line(&mut out, "user", &u.text),
            Turn::Assistant(a) => push_line(&mut out, "assistant", &a.text),
            Turn::Tool(t) => {
                push_line(&mut out, "assistant", &t.assistant.text);
                for (call, result) in &t.calls {
                    push_line(
                        &mut out,
                        "tool_call",
                        &format!("{}({})", call.name, call.args),
                    );
                    push_line(&mut out, "tool_result", &result.content);
                }
            }
        }
    }
    out
}

fn push_line(out: &mut String, role: &str, text: &str) {
    if text.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(role);
    out.push_str(": ");
    out.push_str(text);
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::conversation::{AssistantMsg, ToolResult, ToolTurn};
    use daemon_protocol::UserMsg;

    #[test]
    fn tool_turn_flattens_to_assistant_plus_tool_rows() {
        let tok = Tokenizer::heuristic();
        let turns = vec![Turn::Tool(ToolTurn {
            assistant: AssistantMsg::text("calling"),
            calls: vec![(
                ToolCall {
                    call_id: "c1".into(),
                    name: "fs_read".into(),
                    args: "{}".into(),
                },
                ToolResult {
                    call_id: "c1".into(),
                    ok: true,
                    content: "file body".into(),
                },
            )],
        })];
        let rows = flatten_turns(&turns, &tok);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].role, "assistant");
        assert!(rows[0].tool_calls.as_deref().unwrap().contains("fs_read"));
        assert_eq!(rows[1].role, "tool");
        assert_eq!(rows[1].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(rows[1].tool_name.as_deref(), Some("fs_read"));
    }

    #[test]
    fn render_is_role_tagged() {
        let turns = vec![
            Turn::User(UserMsg::new("hi")),
            Turn::Assistant(AssistantMsg::text("hello")),
        ];
        assert_eq!(render_turns(&turns), "user: hi\nassistant: hello");
    }
}
