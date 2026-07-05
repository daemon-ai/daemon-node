// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Session-text coalescing + the pure-local recap builder (hermes `session_search` / `build_recap`
//! parity).
//!
//! One normalized turn shape ([`IndexTurn`]) is derived from either conversation source — the
//! engine's typed [`Conversation`](daemon_core::Conversation) (the durable snapshot path, full
//! fidelity including tool args) or the wire [`ConvView`](daemon_protocol::ConvView) projection
//! (the live-actor path; tool *names* only, no args) — and feeds three consumers:
//!
//! * [`coalesce_body`] — the searchable FTS body (user + assistant text + tool names) written to
//!   [`SessionStore::index_session_text`](daemon_store::SessionStore::index_session_text) at every
//!   turn boundary (both the durable incarnation and the live event pump index through this).
//! * [`build_recap`] — the node-side `SessionRecap` (hermes `session_recap.py::build_recap`): scope
//!   counts, top tools, recently-touched files, last ask/reply. Pure local, **no LLM call**.
//! * the `session_search` agent tool's read/window shaping (via the same normalized turns).

use daemon_api::SessionRecap;
use daemon_core::{Conversation, Turn};
use daemon_protocol::ConvView;

/// The role of a normalized [`IndexTurn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexRole {
    /// A user message.
    User,
    /// An assistant message (including a tool-calling one).
    Assistant,
}

impl IndexRole {
    /// The lowercase label used in the coalesced body and the tool's JSON shapes.
    pub fn label(self) -> &'static str {
        match self {
            IndexRole::User => "user",
            IndexRole::Assistant => "assistant",
        }
    }
}

/// One conversation turn, normalized across the two conversation sources. A tool-calling assistant
/// turn carries the tool `names` (both sources) and the raw JSON `tool_args` + `tool_results` count
/// (typed-`Conversation` source only; empty/0 from a `ConvView`).
#[derive(Clone, Debug)]
pub struct IndexTurn {
    /// Who spoke.
    pub role: IndexRole,
    /// The turn's text (may be empty for a tool-call-only assistant turn).
    pub text: String,
    /// The names of tools invoked in this turn (empty for non-tool turns).
    pub tools: Vec<String>,
    /// The raw JSON argument payloads of the invoked tools, aligned with [`Self::tools`] where
    /// available (typed-conversation source only; empty from a wire `ConvView`).
    pub tool_args: Vec<String>,
    /// How many tool results this turn recorded (0 when unknown / non-tool).
    pub tool_results: u32,
}

/// Normalize the engine's typed [`Conversation`] (the durable-snapshot source; full fidelity).
pub fn turns_from_conversation(conv: &Conversation) -> Vec<IndexTurn> {
    conv.turns
        .iter()
        .map(|turn| match turn {
            Turn::User(u) => IndexTurn {
                role: IndexRole::User,
                text: u.text.clone(),
                tools: Vec::new(),
                tool_args: Vec::new(),
                tool_results: 0,
            },
            Turn::Assistant(a) => IndexTurn {
                role: IndexRole::Assistant,
                text: a.text.clone(),
                tools: Vec::new(),
                tool_args: Vec::new(),
                tool_results: 0,
            },
            Turn::Tool(t) => IndexTurn {
                role: IndexRole::Assistant,
                text: t.assistant.text.clone(),
                tools: t.calls.iter().map(|(call, _)| call.name.clone()).collect(),
                tool_args: t.calls.iter().map(|(call, _)| call.args.clone()).collect(),
                tool_results: t.calls.len() as u32,
            },
        })
        .collect()
}

/// Normalize a wire [`ConvView`] (the live-actor source; tool names only, and only the last
/// `WIRE_PAGE_MAX` turns — the view is wire-bounded by construction).
pub fn turns_from_view(view: &ConvView) -> Vec<IndexTurn> {
    view.turns
        .iter()
        .map(|t| {
            let role = if t.role == "user" {
                IndexRole::User
            } else {
                // "assistant" and "tool" both render as assistant-authored turns.
                IndexRole::Assistant
            };
            IndexTurn {
                role,
                text: t.text.clone(),
                tools: t.tools.clone(),
                tool_args: Vec::new(),
                tool_results: t.tools.len() as u32,
            }
        })
        .collect()
}

/// The cap on a coalesced FTS body. When the conversation exceeds it, the **oldest** turns are
/// dropped (recent text is the more useful search surface).
pub const INDEX_BODY_CAP: usize = 100_000;

/// Coalesce normalized turns into the searchable FTS body: one `role: text` line per turn
/// (user + assistant text; reasoning is never present in the sources), with tool names appended as
/// `[tools: …]` so "which session used X" queries hit. Tool *result* payloads are deliberately
/// excluded (hermes parity: discovery searches user+assistant text; tool output is noise). Capped
/// at [`INDEX_BODY_CAP`] keeping the tail.
pub fn coalesce_body(turns: &[IndexTurn]) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(turns.len());
    for turn in turns {
        let text = turn.text.trim();
        if text.is_empty() && turn.tools.is_empty() {
            continue;
        }
        let mut line = format!("{}: {}", turn.role.label(), text);
        if !turn.tools.is_empty() {
            line.push_str(&format!(" [tools: {}]", turn.tools.join(" ")));
        }
        lines.push(line);
    }
    // Keep the most recent lines under the cap (drop oldest-first).
    let mut total: usize = lines.iter().map(|l| l.len() + 1).sum();
    let mut start = 0;
    while total > INDEX_BODY_CAP && start < lines.len() {
        total -= lines[start].len() + 1;
        start += 1;
    }
    lines[start..].join("\n")
}

/// How many recent turns the recap treats as "recent activity" (hermes `_RECENT_TURN_WINDOW`; every
/// daemon turn is user- or assistant-authored, so the window is a plain tail slice).
const RECENT_TURN_WINDOW: usize = 20;
/// Preview cap for the latest user prompt (hermes `_PROMPT_PREVIEW_CHARS`).
const PROMPT_PREVIEW_CHARS: usize = 140;
/// Preview cap for the latest assistant text (hermes `_ASSISTANT_PREVIEW_CHARS`).
const ASSISTANT_PREVIEW_CHARS: usize = 200;
/// How many recently-touched files the recap lists (hermes `_MAX_FILES_LISTED`).
const MAX_FILES_LISTED: usize = 5;
/// How many top tools the recap lists.
const MAX_TOOLS_LISTED: usize = 5;

/// Build the pure-local [`SessionRecap`] from normalized turns (hermes
/// `session_recap.py::build_recap`): total scope counts over the whole conversation; top tools,
/// recently-touched files, and the last ask/reply from the recent-activity window. No LLM call.
pub fn build_recap(turns: &[IndexTurn], title: Option<String>) -> SessionRecap {
    let user_turns = turns.iter().filter(|t| t.role == IndexRole::User).count() as u32;
    let assistant_turns = turns
        .iter()
        .filter(|t| t.role == IndexRole::Assistant)
        .count() as u32;
    let tool_results: u32 = turns.iter().map(|t| t.tool_results).sum();

    let window = &turns[turns.len().saturating_sub(RECENT_TURN_WINDOW)..];

    // Tool usage counts over the window, sorted by (-count, name) — hermes `_summarise_tool_activity`.
    let mut counts: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for turn in window {
        for name in &turn.tools {
            *counts.entry(name.as_str()).or_default() += 1;
        }
    }
    let mut top_tools: Vec<(String, u32)> = counts
        .into_iter()
        .map(|(name, count)| (name.to_string(), count))
        .collect();
    top_tools.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top_tools.truncate(MAX_TOOLS_LISTED);

    // Distinct recently-touched files, newest first (walk the window in reverse; `path`/`file_path`
    // string args of any tool call). Args are only present from the typed-conversation source.
    let mut files: Vec<String> = Vec::new();
    for turn in window.iter().rev() {
        for args in turn.tool_args.iter().rev() {
            for path in extract_paths(args) {
                if !files.contains(&path) {
                    files.push(path);
                }
            }
        }
    }
    files.truncate(MAX_FILES_LISTED);

    let last_ask = window
        .iter()
        .rev()
        .filter(|t| t.role == IndexRole::User)
        .map(|t| t.text.trim())
        .find(|t| !t.is_empty())
        .map(|t| truncate_collapsed(t, PROMPT_PREVIEW_CHARS));
    let last_reply = window
        .iter()
        .rev()
        .filter(|t| t.role == IndexRole::Assistant)
        .map(|t| t.text.trim())
        .find(|t| !t.is_empty())
        .map(|t| truncate_collapsed(t, ASSISTANT_PREVIEW_CHARS));

    SessionRecap {
        title,
        user_turns,
        assistant_turns,
        tool_results,
        top_tools,
        files_touched: files,
        last_ask,
        last_reply,
    }
}

/// The `path` / `file_path` string values of a tool-call JSON argument payload (the recap's
/// "files touched" source). Unparseable / non-object args yield nothing.
fn extract_paths(args_json: &str) -> Vec<String> {
    let Ok(serde_json::Value::Object(map)) = serde_json::from_str(args_json) else {
        return Vec::new();
    };
    ["path", "file_path"]
        .iter()
        .filter_map(|key| map.get(*key).and_then(|v| v.as_str()))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Collapse all whitespace runs to single spaces and truncate to `limit` chars with a `…` marker
/// (hermes `_truncate`).
fn truncate_collapsed(text: &str, limit: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= limit {
        return collapsed;
    }
    let head: String = collapsed.chars().take(limit.saturating_sub(1)).collect();
    format!("{}…", head.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::{AssistantMsg, SystemPrompt, ToolCall, ToolResult, ToolTurn};
    use daemon_protocol::{ConvTurnView, UserMsg};

    fn conv() -> Conversation {
        let mut c = Conversation::new(SystemPrompt::new("sys"));
        c.push_user(UserMsg::new("fix the build"));
        c.push_tool(ToolTurn {
            assistant: AssistantMsg::text("let me look"),
            calls: vec![
                (
                    ToolCall {
                        call_id: "c1".into(),
                        name: "fs".into(),
                        args: r#"{"op":"read","path":"src/lib.rs"}"#.into(),
                    },
                    ToolResult {
                        call_id: "c1".into(),
                        ok: true,
                        content: "…".into(),
                    },
                ),
                (
                    ToolCall {
                        call_id: "c2".into(),
                        name: "shell".into(),
                        args: r#"{"command":"cargo build"}"#.into(),
                    },
                    ToolResult {
                        call_id: "c2".into(),
                        ok: true,
                        content: "ok".into(),
                    },
                ),
            ],
        });
        c.push_assistant(AssistantMsg {
            text: "fixed it".into(),
            reasoning: Some("secret chain of thought".into()),
        });
        c
    }

    #[test]
    fn coalesce_includes_user_assistant_and_tool_names_but_not_reasoning_or_results() {
        let body = coalesce_body(&turns_from_conversation(&conv()));
        assert!(body.contains("user: fix the build"));
        assert!(body.contains("assistant: let me look [tools: fs shell]"));
        assert!(body.contains("assistant: fixed it"));
        assert!(!body.contains("secret chain of thought"), "no reasoning");
        assert!(!body.contains("cargo build"), "no tool args/results");
    }

    #[test]
    fn coalesce_caps_body_keeping_the_tail() {
        let turns: Vec<IndexTurn> = (0..200)
            .map(|i| IndexTurn {
                role: IndexRole::User,
                text: format!("turn {i} {}", "x".repeat(1000)),
                tools: Vec::new(),
                tool_args: Vec::new(),
                tool_results: 0,
            })
            .collect();
        let body = coalesce_body(&turns);
        assert!(body.len() <= INDEX_BODY_CAP);
        assert!(!body.contains("turn 0 "), "oldest turns dropped");
        assert!(body.contains("turn 199 "), "newest turns kept");
    }

    #[test]
    fn recap_counts_tools_files_and_last_exchange() {
        let recap = build_recap(
            &turns_from_conversation(&conv()),
            Some("build fixes".into()),
        );
        assert_eq!(recap.title.as_deref(), Some("build fixes"));
        assert_eq!(recap.user_turns, 1);
        assert_eq!(recap.assistant_turns, 2, "tool turn counts as assistant");
        assert_eq!(recap.tool_results, 2);
        assert_eq!(
            recap.top_tools,
            vec![("fs".to_string(), 1), ("shell".to_string(), 1)],
            "ties break by name"
        );
        assert_eq!(recap.files_touched, vec!["src/lib.rs".to_string()]);
        assert_eq!(recap.last_ask.as_deref(), Some("fix the build"));
        assert_eq!(recap.last_reply.as_deref(), Some("fixed it"));
    }

    #[test]
    fn recap_window_limits_tools_and_previews_truncate() {
        // 30 turns: the first 10 use tool "old", the last 20 use tool "new" — only the window counts.
        let mut turns = Vec::new();
        for i in 0..30 {
            let name = if i < 10 { "old" } else { "new" };
            turns.push(IndexTurn {
                role: IndexRole::Assistant,
                text: String::new(),
                tools: vec![name.to_string()],
                tool_args: vec!["{}".into()],
                tool_results: 1,
            });
        }
        turns.push(IndexTurn {
            role: IndexRole::User,
            text: format!("long  ask\n{}", "y".repeat(300)),
            tools: Vec::new(),
            tool_args: Vec::new(),
            tool_results: 0,
        });
        let recap = build_recap(&turns, None);
        assert!(
            recap.top_tools.iter().all(|(name, _)| name == "new"),
            "tools outside the 20-turn window are not counted: {:?}",
            recap.top_tools
        );
        let ask = recap.last_ask.unwrap();
        assert!(ask.chars().count() <= PROMPT_PREVIEW_CHARS);
        assert!(ask.ends_with('…'));
        assert!(ask.starts_with("long ask y"), "whitespace collapsed: {ask}");
        // Totals still cover the WHOLE conversation.
        assert_eq!(recap.tool_results, 30);
    }

    #[test]
    fn view_turns_normalize_roles_and_tool_names() {
        let view = ConvView {
            epoch: 1,
            turns: vec![
                ConvTurnView {
                    role: "user".into(),
                    text: "hi".into(),
                    tools: vec![],
                },
                ConvTurnView {
                    role: "tool".into(),
                    text: "checking".into(),
                    tools: vec!["web_search".into()],
                },
            ],
            waiting_for: vec![],
        };
        let turns = turns_from_view(&view);
        assert_eq!(turns[0].role, IndexRole::User);
        assert_eq!(turns[1].role, IndexRole::Assistant);
        assert_eq!(turns[1].tools, vec!["web_search".to_string()]);
        assert_eq!(turns[1].tool_results, 1);
        let recap = build_recap(&turns, None);
        assert!(
            recap.files_touched.is_empty(),
            "no args from a wire view -> no files"
        );
    }

    #[test]
    fn empty_conversation_recaps_to_zeroes() {
        let recap = build_recap(&[], None);
        assert_eq!(recap.user_turns, 0);
        assert!(recap.last_ask.is_none() && recap.last_reply.is_none());
        assert!(recap.top_tools.is_empty() && recap.files_touched.is_empty());
    }
}
