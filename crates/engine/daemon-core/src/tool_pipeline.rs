//! The tool execution pipeline (§12 `run_tool`).
//!
//! Resolves a call against the registry, runs it against the turn context (where the tool itself
//! enforces its own preflight safety — workspace containment via the [`ExecutionEnvironment`](crate::exec),
//! and hardline-deny + interactive `HostRequest::Approval` for command execution), then applies the
//! cross-cutting **sanitize + result-byte budget** stage uniformly so one oversized tool result can
//! never blow the model context.
//!
//! Deferred (later slices, noted so the seam is explicit): arg-JSON repair, the checkpoint-if-mutating
//! stage (needs the store/git), untrusted-output wrapping for web/MCP sources, and parallel tool
//! batching (tools run sequentially here). An unknown tool surfaces a failed result, never a panic.

use crate::conversation::{ToolCall, ToolResult};
use crate::tools::{ToolOutcome, ToolRegistry};
use crate::turn::TurnCx;

/// Run one tool call through the pipeline (§12): resolve -> execute -> sanitize + budget.
pub async fn run_tool(call: &ToolCall, registry: &ToolRegistry, cx: &TurnCx<'_>) -> ToolOutcome {
    let mut outcome = match registry.get(&call.name) {
        Some(tool) => tool.run(call, cx).await,
        None => ToolOutcome::text(
            call.call_id.clone(),
            false,
            format!("unknown tool: {}", call.name),
        ),
    };
    budget_result(&mut outcome.result, cx.tool_result_budget);
    outcome
}

/// The §12 sanitize+budget stage: truncate an oversized tool result to the per-tool budget, leaving
/// a clear marker, so a single large result cannot dominate the model context. `0` disables the cap.
fn budget_result(result: &mut ToolResult, budget: usize) {
    if budget == 0 || result.content.len() <= budget {
        return;
    }
    let mut cut = budget.min(result.content.len());
    while cut > 0 && !result.content.is_char_boundary(cut) {
        cut -= 1;
    }
    let dropped = result.content.len() - cut;
    result.content.truncate(cut);
    result
        .content
        .push_str(&format!("\n... [truncated {dropped} bytes over result budget]"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_truncates_oversized_results_and_marks_them() {
        let mut result = ToolResult {
            call_id: "c1".into(),
            ok: true,
            content: "x".repeat(1000),
        };
        budget_result(&mut result, 100);
        assert!(result.content.starts_with(&"x".repeat(100)));
        assert!(result.content.contains("truncated"));
        assert!(result.content.contains("900 bytes"));
    }

    #[test]
    fn budget_zero_disables_truncation() {
        let mut result = ToolResult {
            call_id: "c1".into(),
            ok: true,
            content: "x".repeat(1000),
        };
        budget_result(&mut result, 0);
        assert_eq!(result.content.len(), 1000);
        assert!(!result.content.contains("truncated"));
    }

    #[test]
    fn budget_respects_char_boundaries() {
        // A multi-byte char straddling the cut point must not split mid-codepoint (no panic).
        let mut result = ToolResult {
            call_id: "c1".into(),
            ok: true,
            content: "é".repeat(100),
        };
        budget_result(&mut result, 51);
        assert!(result.content.contains("truncated"));
    }
}
