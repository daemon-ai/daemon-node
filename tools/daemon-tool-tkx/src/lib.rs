// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-tkx` — agent-managed task/knowledge tooling implementing `daemon_core::Tool`.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
use serde::Deserialize;

/// The JSON-Schema advertised for the `tkx` tool.
const TKX_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["status"],
      "description": "Only `status` is available until the task/knowledge tracker backend is implemented."
    }
  }
}"#;

/// The task/knowledge tracker tool.
///
/// `tkx` is intentionally present in the workspace because orchestration and lifecycle specs use it
/// as the canonical tool-owned work source. Until the backing tracker lands, the tool fails closed with
/// a clear status instead of being an empty crate or a silent no-op.
#[derive(Default)]
pub struct TkxTool;

impl TkxTool {
    /// A `tkx` tool with no backend attached.
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    #[serde(default = "default_action")]
    action: String,
}

fn default_action() -> String {
    "status".to_string()
}

#[async_trait]
impl Tool for TkxTool {
    fn name(&self) -> &str {
        "tkx"
    }

    fn schema(&self) -> &str {
        TKX_SCHEMA
    }

    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let args: Args = match serde_json::from_str(&call.args) {
            Ok(a) => a,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("tkx: invalid arguments: {e}"),
                )
            }
        };

        match args.action.as_str() {
            "status" => ToolOutcome::text(
                call.call_id.clone(),
                false,
                "tkx: tracker backend is not implemented yet; use the `todo` tool for in-turn task state.",
            ),
            other => ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!("tkx: unsupported action {other:?}; only \"status\" is available"),
            ),
        }
    }
}
