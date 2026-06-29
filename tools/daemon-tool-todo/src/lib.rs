// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-todo` — the `todo` chat tool (`daemon_core::Tool`): an agent-managed, per-session
//! task list (the structured planning surface a GUI renders in its status stack).
//!
//! The tool keeps the current list per `session_id`. A call either replaces the list (`merge =
//! false`, the default) or upserts items by `id` (`merge = true`), mirroring the
//! create-then-update-in-place planning loop. The rendered checklist is returned for the model and a
//! structured `ToolDetail { kind: "todo" }` payload is attached for the GUI. The list is agent
//! state, not external content, so it is **not** marked untrusted.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use daemon_common::SessionId;
use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
use daemon_protocol::ToolDetail;
use serde::{Deserialize, Serialize};

/// The JSON-Schema advertised for the `todo` tool.
const TODO_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["todos"],
  "properties": {
    "todos": {
      "type": "array",
      "description": "The task list. Replaces the current list unless `merge` is true.",
      "items": {
        "type": "object",
        "required": ["id", "content", "status"],
        "properties": {
          "id": {"type": "string", "description": "Stable identifier for the task."},
          "content": {"type": "string", "description": "The task description."},
          "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]}
        }
      }
    },
    "merge": {"type": "boolean", "description": "Upsert items by id instead of replacing the list (default false)."}
  }
}"#;

/// A task's lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Not started.
    Pending,
    /// Actively being worked on.
    InProgress,
    /// Finished successfully.
    Completed,
    /// No longer needed.
    Cancelled,
}

impl TodoStatus {
    /// A checkbox-style glyph for the rendered list.
    fn glyph(self) -> &'static str {
        match self {
            TodoStatus::Pending => "[ ]",
            TodoStatus::InProgress => "[~]",
            TodoStatus::Completed => "[x]",
            TodoStatus::Cancelled => "[-]",
        }
    }
}

/// One task in the list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TodoItem {
    /// Stable identifier (the merge key).
    pub id: String,
    /// The task description.
    pub content: String,
    /// The lifecycle state.
    pub status: TodoStatus,
}

/// The `todo` tool. Holds the per-session task list behind a mutex (it is shared across sessions via
/// `Arc`, but each session's list is keyed by `session_id`).
#[derive(Default)]
pub struct TodoTool {
    lists: Mutex<HashMap<SessionId, Vec<TodoItem>>>,
}

impl TodoTool {
    /// A todo tool with no lists yet.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    todos: Vec<TodoItem>,
    #[serde(default)]
    merge: bool,
}

#[async_trait]
impl Tool for TodoTool {
    fn name(&self) -> &str {
        "todo"
    }

    fn schema(&self) -> &str {
        TODO_SCHEMA
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let args: Args = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("todo: invalid arguments: {e}"),
                )
            }
        };

        let list = {
            let mut lists = self.lists.lock().expect("todo lock poisoned");
            let current = lists.entry(cx.session_id.clone()).or_default();
            if args.merge {
                upsert(current, args.todos);
            } else {
                *current = args.todos;
            }
            current.clone()
        };

        let content = render(&list);
        let detail = ToolDetail {
            kind: "todo".to_string(),
            body: serde_json::to_vec(&list).unwrap_or_default(),
        };
        ToolOutcome::text(call.call_id.clone(), true, content).with_detail(detail)
    }
}

/// Upsert `incoming` items into `current` by id (updating in place, appending new ones).
fn upsert(current: &mut Vec<TodoItem>, incoming: Vec<TodoItem>) {
    for item in incoming {
        if let Some(existing) = current.iter_mut().find(|c| c.id == item.id) {
            *existing = item;
        } else {
            current.push(item);
        }
    }
}

/// Render the list as a compact checklist for the model.
fn render(list: &[TodoItem]) -> String {
    if list.is_empty() {
        return "todo: list cleared (no items)".to_string();
    }
    let done = list
        .iter()
        .filter(|i| i.status == TodoStatus::Completed)
        .count();
    let mut out = format!("todo: {}/{} complete\n", done, list.len());
    for item in list {
        out.push_str(&format!("{} {}\n", item.status.glyph(), item.content));
    }
    out
}
