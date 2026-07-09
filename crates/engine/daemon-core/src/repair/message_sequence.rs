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

    pub(super) fn user(text: &str) -> RequestMsg {
        RequestMsg {
            role: "user".into(),
            content: text.into(),
            ..Default::default()
        }
    }
    pub(super) fn assistant(text: &str) -> RequestMsg {
        RequestMsg {
            role: "assistant".into(),
            content: text.into(),
            ..Default::default()
        }
    }
    pub(super) fn assistant_calls(text: &str, ids: &[&str]) -> RequestMsg {
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
    pub(super) fn tool(id: &str, content: &str) -> RequestMsg {
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

/// Parity tests ported from hermes' `repair_message_sequence`
/// (`agent/agent_runtime_helpers.py:347`; tests
/// `tests/run_agent/test_message_sequence_repair.py`). The Python pass drops stray tool results
/// (already handled by the Rust port) and merges consecutive `user` messages (the gap).
///
/// Adaptation: hermes represents multimodal user content as a *list* and skips merging when a side
/// is a list; the Rust `RequestMsg` carries text in `content` and images in `images`, so the Rust
/// port skips merging when either side has non-empty `images`. hermes returns a repair count; the
/// Rust port returns the repaired `Vec`, so parity is asserted on the resulting messages.
///
/// `parity_gap_*` tests assert the desired Python behavior and are expected to FAIL until merging is
/// ported; plain-named tests port behavior the Rust port already has and MUST PASS.
#[cfg(test)]
mod parity {
    use super::tests::{assistant, assistant_calls, tool, user};
    use super::*;
    use crate::provider::RequestImage;

    fn user_multimodal(text: &str) -> RequestMsg {
        RequestMsg {
            role: "user".into(),
            content: text.into(),
            images: vec![RequestImage {
                mime: "image/png".into(),
                data_base64: "AAAA".into(),
            }],
            ..Default::default()
        }
    }

    // ── Consecutive-user merge (gap) ──────────────────────────────────────

    // parity: test_message_sequence_repair.py::test_repair_merges_consecutive_user_messages (tests/run_agent/test_message_sequence_repair.py:81)
    #[test]
    fn parity_gap_merges_consecutive_user_messages() {
        let out = repair_message_sequence(vec![user("first"), user("second")]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "user");
        assert_eq!(out[0].content, "first\n\nsecond");
    }

    // ── Behavior the Rust port already has (must PASS) ────────────────────

    // parity: test_message_sequence_repair.py::test_repair_preserves_user_content_when_one_side_empty (tests/run_agent/test_message_sequence_repair.py:96)
    #[test]
    fn preserves_user_content_when_one_side_empty() {
        let out = repair_message_sequence(vec![user(""), user("real message")]);
        assert_eq!(out, vec![user("real message")]);
    }

    // parity: test_message_sequence_repair.py::test_repair_preserves_multimodal_user_content (tests/run_agent/test_message_sequence_repair.py:165)
    #[test]
    fn preserves_multimodal_user_content() {
        // A user message with images must NOT be merged into an adjacent user message.
        let out = repair_message_sequence(vec![user_multimodal("hi"), user("follow-up")]);
        assert_eq!(out.len(), 2);
        assert!(!out[0].images.is_empty());
    }

    // parity: test_message_sequence_repair.py::test_repair_does_not_rewind_ongoing_dialog_tool_pair (tests/run_agent/test_message_sequence_repair.py:108)
    #[test]
    fn does_not_rewind_ongoing_dialog_tool_pair() {
        // assistant(tool_calls) + tool + user is a VALID pattern (user redirect before the model's
        // continuation turn); repair must leave it untouched.
        let input = vec![
            user("Q1"),
            assistant_calls("", &["t1"]),
            tool("t1", "out"),
            user("Q2"),
        ];
        assert_eq!(repair_message_sequence(input.clone()), input);
    }

    // parity: test_message_sequence_repair.py::test_repair_drops_stray_tool_with_unknown_tool_call_id (tests/run_agent/test_message_sequence_repair.py:131)
    #[test]
    fn drops_stray_tool_with_unknown_tool_call_id() {
        let out = repair_message_sequence(vec![
            user("hi"),
            assistant("hello"),
            tool("orphan", "stray"),
            user("real"),
        ]);
        assert!(out.iter().all(|m| m.role != "tool"));
    }

    // parity: test_message_sequence_repair.py::test_repair_leaves_valid_conversation_unchanged (tests/run_agent/test_message_sequence_repair.py:146)
    #[test]
    fn leaves_valid_conversation_unchanged() {
        let input = vec![
            user("list files"),
            assistant_calls("", &["t1"]),
            tool("t1", "a.txt b.txt"),
            assistant("Found 2 files"),
            user("more"),
        ];
        assert_eq!(repair_message_sequence(input.clone()), input);
    }

    // parity: test_message_sequence_repair.py::test_repair_empty_messages_returns_zero (tests/run_agent/test_message_sequence_repair.py:181)
    #[test]
    fn empty_messages_is_noop() {
        assert_eq!(repair_message_sequence(vec![]), vec![]);
    }

    // NOTE: test_repair_preserves_system_messages is out-of-scope — in the Rust engine the system
    // prompt is a separate wire field, not part of this flattened message sequence, so the leading
    // non-`user` trim intentionally drops a leading `system` message (see PARITY.md).
}
