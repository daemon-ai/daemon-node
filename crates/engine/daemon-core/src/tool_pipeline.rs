//! The tool execution pipeline (§12 `run_tool`).
//!
//! Resolves a call against the registry and runs it against the turn context. The full §12 pipeline
//! (arg repair/validation, preflight safety + approval gate, checkpoint-if-mutating, sandbox
//! execution, sanitize + wrap-untrusted + budget) layers onto this seam in later slices; phase 3
//! implements resolution + execution and surfaces an unknown-tool result rather than panicking.

use crate::conversation::{ToolCall, ToolResult};
use crate::tools::{ToolOutcome, ToolRegistry};
use crate::turn::TurnCx;

/// Run one tool call through the pipeline (§12).
pub async fn run_tool(call: &ToolCall, registry: &ToolRegistry, cx: &TurnCx<'_>) -> ToolOutcome {
    match registry.get(&call.name) {
        Some(tool) => tool.run(call, cx).await,
        None => ToolOutcome {
            result: ToolResult {
                call_id: call.call_id.clone(),
                ok: false,
                content: format!("unknown tool: {}", call.name),
            },
            effects: Vec::new(),
        },
    }
}
