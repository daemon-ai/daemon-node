// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Wire message-sequence repair (§9, the flatten boundary).
//!
//! The typed [`Conversation`](crate::conversation) keeps a tool call and its result slot together in
//! a [`ToolTurn`](crate::conversation::ToolTurn), so the engine never produces an orphaned pair
//! *internally* (hermes' preflight conversation sanitizer is unnecessary at that level — see
//! [`super`]). What still needs a safety net is the **flattened wire sequence** handed to a provider:
//! Turn-granular compaction (§10) can drop a leading user turn (leaving an assistant/tool message
//! first, which Anthropic rejects), and a turn suspended mid-tool can flatten to a tool call whose
//! result slot is still empty. This pass enforces the structural contract every chat provider (and
//! Anthropic strictly) expects, mirroring hermes `agent/message_sanitization.py`:
//!
//! 1. **Leading-user**: drop leading non-`user` messages so the first message is a user message.
//! 2. **Tool pairing**: every assistant `tool_calls` entry gets a matching `tool` result; a missing
//!    one is back-filled with a synthetic placeholder (orphaned tool call), and a `tool` result with
//!    no matching call is dropped (orphaned tool result).
//! 3. **No empties**: a plain `user`/`assistant` message with blank content and no tool calls is
//!    dropped, and an empty tool-result body is coerced to a placeholder (providers reject empty
//!    content blocks).
//!
//! It is a no-op for a well-formed sequence (the common case), so it never perturbs the normal path.

use std::collections::HashSet;

use crate::provider::RequestMsg;

/// The placeholder used for a tool call whose result is missing/empty (so the provider sees a
/// complete, well-formed tool exchange even when a turn was suspended or interrupted mid-call).
const MISSING_RESULT: &str = "[tool result unavailable: the call did not complete]";

/// Repair a flattened wire message sequence to satisfy provider structural contracts. See the module
/// docs for the rules. No-op for an already well-formed sequence.
pub fn repair_message_sequence(mut messages: Vec<RequestMsg>) -> Vec<RequestMsg> {
    // (1) Leading-user: drop everything before the first user message. A leading assistant/tool
    //     message cannot open a valid turn; dropping it also discards results it would orphan.
    if let Some(first_user) = messages.iter().position(|m| m.role == "user") {
        if first_user > 0 {
            messages.drain(0..first_user);
        }
    }

    let mut out: Vec<RequestMsg> = Vec::with_capacity(messages.len());
    let mut i = 0;
    while i < messages.len() {
        let role = messages[i].role.as_str();
        if role == "assistant" && !messages[i].tool_calls.is_empty() {
            // (2) An assistant message that issued tool calls: emit it, then reconcile the contiguous
            //     run of tool results that follows against the calls it expects.
            let expected: Vec<String> = messages[i]
                .tool_calls
                .iter()
                .map(|c| c.call_id.clone())
                .collect();
            out.push(messages[i].clone());
            i += 1;

            let mut answered: HashSet<String> = HashSet::new();
            while i < messages.len() && messages[i].role == "tool" {
                let mut result = messages[i].clone();
                let id = result.tool_call_id.clone().unwrap_or_default();
                // Keep the first result for an expected call; drop orphans and duplicates.
                if expected.contains(&id) && answered.insert(id) {
                    if result.content.trim().is_empty() {
                        result.content = MISSING_RESULT.to_string();
                    }
                    out.push(result);
                }
                i += 1;
            }
            // Back-fill a synthetic result for any expected call left unanswered, in call order.
            for id in &expected {
                if !answered.contains(id) {
                    out.push(RequestMsg {
                        role: "tool".into(),
                        content: MISSING_RESULT.to_string(),
                        tool_call_id: Some(id.clone()),
                        ..Default::default()
                    });
                }
            }
        } else if role == "tool" {
            // (2) A tool result with no preceding assistant tool call in this position: orphan, drop.
            i += 1;
        } else if role == "assistant" || role == "user" {
            // (3) Drop empty plain messages; keep anything with content.
            if !messages[i].content.trim().is_empty() {
                out.push(messages[i].clone());
            }
            i += 1;
        } else {
            out.push(messages[i].clone());
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::ToolCall;

    fn user(text: &str) -> RequestMsg {
        RequestMsg {
            role: "user".into(),
            content: text.into(),
            ..Default::default()
        }
    }
    fn assistant(text: &str) -> RequestMsg {
        RequestMsg {
            role: "assistant".into(),
            content: text.into(),
            ..Default::default()
        }
    }
    fn assistant_calls(text: &str, ids: &[&str]) -> RequestMsg {
        RequestMsg {
            role: "assistant".into(),
            content: text.into(),
            tool_calls: ids
                .iter()
                .map(|id| ToolCall {
                    call_id: (*id).into(),
                    name: "t".into(),
                    args: "{}".into(),
                })
                .collect(),
            ..Default::default()
        }
    }
    fn tool(id: &str, content: &str) -> RequestMsg {
        RequestMsg {
            role: "tool".into(),
            content: content.into(),
            tool_call_id: Some(id.into()),
            ..Default::default()
        }
    }

    #[test]
    fn well_formed_sequence_is_unchanged() {
        let input = vec![
            user("hi"),
            assistant_calls("calling", &["a"]),
            tool("a", "result a"),
            assistant("done"),
        ];
        assert_eq!(repair_message_sequence(input.clone()), input);
    }

    #[test]
    fn leading_non_user_is_trimmed() {
        let out = repair_message_sequence(vec![assistant("stray"), user("hi")]);
        assert_eq!(out, vec![user("hi")]);
    }

    #[test]
    fn orphaned_tool_call_gets_synthetic_result() {
        let out = repair_message_sequence(vec![
            user("hi"),
            assistant_calls("calling", &["a", "b"]),
            tool("a", "ok"),
        ]);
        // b had no result -> synthesized.
        assert_eq!(out.len(), 4);
        assert_eq!(out[3].role, "tool");
        assert_eq!(out[3].tool_call_id.as_deref(), Some("b"));
        assert!(out[3].content.contains("unavailable"));
    }

    #[test]
    fn orphaned_tool_result_is_dropped() {
        let out = repair_message_sequence(vec![
            user("hi"),
            tool("ghost", "no call"),
            assistant("done"),
        ]);
        assert_eq!(out, vec![user("hi"), assistant("done")]);
    }

    #[test]
    fn empty_tool_result_is_coerced() {
        let out = repair_message_sequence(vec![
            user("hi"),
            assistant_calls("c", &["a"]),
            tool("a", "  "),
        ]);
        assert_eq!(out[2].content, MISSING_RESULT);
    }

    #[test]
    fn empty_plain_messages_are_dropped() {
        let out = repair_message_sequence(vec![user("hi"), assistant(""), assistant("real")]);
        assert_eq!(out, vec![user("hi"), assistant("real")]);
    }

    #[test]
    fn duplicate_tool_result_is_dropped() {
        let out = repair_message_sequence(vec![
            user("hi"),
            assistant_calls("c", &["a"]),
            tool("a", "first"),
            tool("a", "dup"),
        ]);
        assert_eq!(out.iter().filter(|m| m.role == "tool").count(), 1);
        assert_eq!(out[2].content, "first");
    }
}
