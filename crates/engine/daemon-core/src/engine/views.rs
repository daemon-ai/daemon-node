// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Pure, stateless helpers for the engine turn loop: the ?4.3 effect router and the ?17 transcript
//! view + ?4.2 round-signature + failure/recovery classification functions.
//!
//! Split out of `engine.rs` so the module body is the `Engine` turn machinery; these field-free
//! helpers (none touch `Engine` state) lived alongside it only for proximity. Behavior-preserving:
//! every item is the verbatim move from `engine.rs`.

use crate::recovery::RecoveryStep;
use crate::turn::Effect;
use crate::Failure;
use daemon_common::JobId;
use daemon_protocol::{ToolCallView, ToolDetail, ToolResultView};
use std::time::Duration;

use crate::conversation::Turn;

/// Map a config millisecond timeout to the pipeline's `Option<Duration>`: `0` ? `None` (the ?12
/// per-tool timeout stage is disabled), any positive value ? `Some(Duration)`.
pub(super) fn timeout_from_ms(ms: u64) -> Option<Duration> {
    (ms > 0).then(|| Duration::from_millis(ms))
}

/// The single-owner applier's view of a tool batch's [`Effect`]s, partitioned by kind (?4.3): the
/// `Persist` turns to append to the conversation, the at-most-one `Delegate` (job + payload) that
/// drives suspension, the fire-and-forget `Spawn`s, and the `AwaitDecision` parks (?12 HITL). Pure:
/// the caller replays `persists` and acts on the rest, so no conversation mutation hides in here.
pub(super) struct PartitionedEffects {
    /// Turns to append to the conversation (durable record), in effect order.
    pub(super) persists: Vec<Turn>,
    /// The delegated job + opaque payload to suspend on, if the batch delegated (last one wins,
    /// matching the original inline router).
    pub(super) delegated: Option<(JobId, Vec<u8>)>,
    /// Attached, non-joining background children to spawn fire-and-forget.
    pub(super) spawns: Vec<daemon_protocol::SpawnSpec>,
    /// Gated tool calls awaiting a durable operator decision.
    pub(super) awaiting: Vec<crate::snapshot::PendingApproval>,
}

/// Partition a tool batch's [`Effect`]s into [`PartitionedEffects`] (?4.3 effect router). Verbatim
/// of the original inline `match`, but pure: `Persist` turns are collected (not pushed) so the
/// single-owner applier remains the sole mutator of the conversation.
pub(super) fn partition_tool_effects(effects: Vec<Effect>) -> PartitionedEffects {
    let mut persists: Vec<Turn> = Vec::new();
    let mut delegated: Option<(JobId, Vec<u8>)> = None;
    let mut spawns: Vec<daemon_protocol::SpawnSpec> = Vec::new();
    let mut awaiting: Vec<crate::snapshot::PendingApproval> = Vec::new();
    for effect in effects {
        match effect {
            Effect::Persist(turn) => persists.push(turn),
            Effect::Delegate { job, payload } => delegated = Some((job, payload)),
            Effect::Spawn(spec) => spawns.push(spec),
            Effect::AwaitDecision {
                job_id,
                call,
                prompt,
                path,
            } => awaiting.push(crate::snapshot::PendingApproval {
                job_id,
                call,
                prompt,
                path,
                // The approval fingerprint (Cluster B) is stamped by the engine at park time via
                // `Tool::resolved_fingerprint` (it needs the tool registry + cx, which this pure
                // partitioner does not have). `Effect::AwaitDecision` stays unchanged so the shared
                // variant does not force edits on other tools (e.g. execute_code).
                fingerprint: None,
            }),
        }
    }
    PartitionedEffects {
        persists,
        delegated,
        spawns,
        awaiting,
    }
}

/// The ?17 transcript view of a tool *call* (`ToolStarted`): name + a generic structured echo of
/// the call arguments, opaque to the daemon. A tool with a richer call schema can refine `detail`
/// once providers carry structured args.
pub(super) fn tool_call_view(call: &crate::conversation::ToolCall) -> ToolCallView {
    ToolCallView {
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        args_summary: call.args.clone(),
        detail: Some(ToolDetail {
            kind: call.name.clone(),
            body: call.args.clone().into_bytes(),
        }),
    }
}

/// The ?17 transcript view of a tool *result* (`ToolFinished`): the summary text plus the tool's
/// typed output (`detail`) for a rich consumer; `detail` is `None` for plain-text tools.
pub(super) fn tool_result_view(outcome: &crate::tools::ToolOutcome) -> ToolResultView {
    ToolResultView {
        call_id: outcome.result.call_id.clone(),
        ok: outcome.result.ok,
        summary: outcome.result.content.clone(),
        detail: outcome.detail.clone(),
    }
}

/// The ?17 transcript view of a bare [`ToolResult`](crate::conversation::ToolResult) with no typed
/// detail ? used for a guardrail-**blocked** call whose synthetic error result never produced a
/// [`ToolOutcome`](crate::tools::ToolOutcome).
pub(super) fn tool_result_view_of(result: &crate::conversation::ToolResult) -> ToolResultView {
    ToolResultView {
        call_id: result.call_id.clone(),
        ok: result.ok,
        summary: result.content.clone(),
        detail: None,
    }
}

/// Whether a model-emitted tool batch is safe to run concurrently (hermes
/// `_should_parallelize_tool_batch`). Every call must resolve to a tool whose **per-call** class is
/// [`ToolConcurrency::Parallel`](crate::tools::ToolConcurrency), and no two path-scoped calls may
/// overlap (prefix-subtree). An unresolved tool, any exclusive call, or an overlapping pair forces
/// sequential execution. A tool that declares no path scope (`parallel_scope_paths == None`) imposes
/// no overlap constraint (a read-only tool like `web_search`); path-scoped tools reserve their paths.
pub(super) fn batch_is_parallelizable(
    tool_calls: &[crate::conversation::ToolCall],
    registry: &crate::tools::ToolRegistry,
) -> bool {
    let mut reserved: Vec<std::path::PathBuf> = Vec::new();
    for call in tool_calls {
        let Some(tool) = registry.get(&call.name) else {
            return false;
        };
        if tool.concurrency_for(call) != crate::tools::ToolConcurrency::Parallel {
            return false;
        }
        if let Some(paths) = tool.parallel_scope_paths(call) {
            for path in paths {
                if reserved.iter().any(|other| paths_overlap(other, &path)) {
                    return false;
                }
                reserved.push(path);
            }
        }
    }
    true
}

/// Whether two paths may refer to the same subtree (hermes `_paths_overlap`): true when one is a
/// prefix of the other over path components (ancestor/descendant/equal). Siblings do not overlap; an
/// empty path never overlaps.
pub(super) fn paths_overlap(left: &std::path::Path, right: &std::path::Path) -> bool {
    let left: Vec<_> = left.components().collect();
    let right: Vec<_> = right.components().collect();
    if left.is_empty() || right.is_empty() {
        return false;
    }
    let common = left.len().min(right.len());
    left[..common] == right[..common]
}

/// A stable hash of one tool round's calls and results (?4.2 no-progress guard). Hashes each call's
/// `name` + `args` and its result's `ok` + `content`, but **not** the per-call `call_id` (freshly
/// minted each round), so two rounds that issue the same calls and get the same results hash equal —
/// the signal that the model is looping without converging.
pub(super) fn round_signature(
    calls: &[(
        crate::conversation::ToolCall,
        crate::conversation::ToolResult,
    )],
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    calls.len().hash(&mut hasher);
    for (call, result) in calls {
        call.name.hash(&mut hasher);
        call.args.hash(&mut hasher);
        result.ok.hash(&mut hasher);
        result.content.hash(&mut hasher);
    }
    hasher.finish()
}

/// A static label for a [`Failure`] kind, for structured `tracing` of the ?8 recovery loop.
pub(super) fn failure_kind(failure: &Failure) -> &'static str {
    match failure {
        Failure::Provider(_) => "provider",
        Failure::Rotatable(_) => "rotatable",
        Failure::RateLimit { .. } => "rate_limit",
        Failure::Billing(_) => "billing",
        Failure::Auth(_) => "auth",
        Failure::ContextOverflow(_) => "context_overflow",
        Failure::PayloadTooLarge(_) => "payload_too_large",
        Failure::ContentPolicy(_) => "content_policy",
        Failure::FormatError(_) => "format_error",
        Failure::TransientTransport(_) => "transient_transport",
        Failure::ProviderOverloaded(_) => "provider_overloaded",
        Failure::Fatal(_) => "fatal",
        Failure::Cancelled => "cancelled",
        Failure::Other(_) => "other",
    }
}

/// A static label for a [`RecoveryStep`] decision, for structured `tracing` of the ?8 recovery loop.
pub(super) fn recovery_step_kind(step: &RecoveryStep) -> &'static str {
    match step {
        RecoveryStep::Retry { .. } => "retry",
        RecoveryStep::Rotate => "rotate",
        RecoveryStep::Compact => "compact",
        RecoveryStep::Fallback => "fallback",
        RecoveryStep::Abort => "abort",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::ToolCall;
    use crate::tools::{Tool, ToolConcurrency, ToolOutcome, ToolRegistry};
    use crate::turn::TurnCx;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;

    /// A configurable test tool: fixed per-call concurrency class + optional declared paths (read
    /// from the call's `path`/`paths` args), so `batch_is_parallelizable` can be exercised directly.
    struct FakeTool {
        name: &'static str,
        class: ToolConcurrency,
        path_scoped: bool,
    }

    #[async_trait::async_trait]
    impl Tool for FakeTool {
        fn name(&self) -> &str {
            self.name
        }
        fn schema(&self) -> &str {
            "{}"
        }
        fn concurrency(&self) -> ToolConcurrency {
            self.class
        }
        fn parallel_scope_paths(&self, call: &ToolCall) -> Option<Vec<PathBuf>> {
            if !self.path_scoped {
                return None;
            }
            let v: serde_json::Value = serde_json::from_str(&call.args).ok()?;
            let p = v.get("path")?.as_str()?;
            Some(vec![PathBuf::from(p)])
        }
        async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
            ToolOutcome::text(call.call_id.clone(), true, "ok")
        }
    }

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            call_id: format!("c-{name}"),
            name: name.to_string(),
            args: args.to_string(),
        }
    }

    #[test]
    fn timeout_from_ms_zero_is_none() {
        assert_eq!(timeout_from_ms(0), None);
        assert_eq!(timeout_from_ms(1500), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn paths_overlap_prefix_subtree() {
        // Equal / ancestor / descendant overlap; siblings do not; empty never does.
        assert!(paths_overlap(Path::new("/a/b"), Path::new("/a/b")));
        assert!(paths_overlap(Path::new("/a/b"), Path::new("/a/b/c")));
        assert!(paths_overlap(Path::new("/a"), Path::new("/a/b/c")));
        assert!(!paths_overlap(Path::new("/a/b"), Path::new("/a/c")));
        assert!(!paths_overlap(Path::new("/a/b"), Path::new("/x/y")));
        assert!(!paths_overlap(Path::new(""), Path::new("/a")));
    }

    fn registry() -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(FakeTool {
            name: "read",
            class: ToolConcurrency::Parallel,
            path_scoped: true,
        }));
        r.register(Arc::new(FakeTool {
            name: "web",
            class: ToolConcurrency::Parallel,
            path_scoped: false,
        }));
        r.register(Arc::new(FakeTool {
            name: "write",
            class: ToolConcurrency::Exclusive,
            path_scoped: true,
        }));
        r
    }

    #[test]
    fn parallelizable_all_parallel_no_paths() {
        let reg = registry();
        assert!(batch_is_parallelizable(
            &[call("web", "{}"), call("web", "{}")],
            &reg
        ));
    }

    #[test]
    fn not_parallelizable_with_exclusive() {
        let reg = registry();
        assert!(!batch_is_parallelizable(
            &[call("web", "{}"), call("write", "{\"path\":\"a\"}")],
            &reg
        ));
    }

    #[test]
    fn not_parallelizable_unknown_tool() {
        let reg = registry();
        assert!(!batch_is_parallelizable(
            &[call("web", "{}"), call("ghost", "{}")],
            &reg
        ));
    }

    #[test]
    fn path_scoped_disjoint_parallel_overlap_serial() {
        let reg = registry();
        // Disjoint read paths ? parallel.
        assert!(batch_is_parallelizable(
            &[
                call("read", "{\"path\":\"/w/a\"}"),
                call("read", "{\"path\":\"/w/b\"}"),
            ],
            &reg
        ));
        // Overlapping (ancestor/descendant) read paths ? serial.
        assert!(!batch_is_parallelizable(
            &[
                call("read", "{\"path\":\"/w/a\"}"),
                call("read", "{\"path\":\"/w/a/deep\"}"),
            ],
            &reg
        ));
    }
}
