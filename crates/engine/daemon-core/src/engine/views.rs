// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Pure, stateless helpers for the engine turn loop: the §4.3 effect router and the §17 transcript
//! view + §4.2 round-signature + failure/recovery classification functions.
//!
//! Split out of `engine.rs` so the module body is the `Engine` turn machinery; these field-free
//! helpers (none touch `Engine` state) lived alongside it only for proximity. Behavior-preserving:
//! every item is the verbatim move from `engine.rs`.

use crate::recovery::RecoveryStep;
use crate::turn::Effect;
use crate::Failure;
use daemon_common::JobId;
use daemon_protocol::{ToolCallView, ToolDetail, ToolResultView};

use crate::conversation::Turn;

/// The single-owner applier's view of a tool batch's [`Effect`]s, partitioned by kind (§4.3): the
/// `Persist` turns to append to the conversation, the at-most-one `Delegate` (job + payload) that
/// drives suspension, the fire-and-forget `Spawn`s, and the `AwaitDecision` parks (§12 HITL). Pure:
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

/// Partition a tool batch's [`Effect`]s into [`PartitionedEffects`] (§4.3 effect router). Verbatim
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

/// The §17 transcript view of a tool *call* (`ToolStarted`): name + a generic structured echo of
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

/// The §17 transcript view of a tool *result* (`ToolFinished`): the summary text plus the tool's
/// typed output (`detail`) for a rich consumer; `detail` is `None` for plain-text tools.
pub(super) fn tool_result_view(outcome: &crate::tools::ToolOutcome) -> ToolResultView {
    ToolResultView {
        call_id: outcome.result.call_id.clone(),
        ok: outcome.result.ok,
        summary: outcome.result.content.clone(),
        detail: outcome.detail.clone(),
    }
}

/// A stable hash of one tool round's calls and results (§4.2 no-progress guard). Hashes each call's
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

/// A static label for a [`Failure`] kind, for structured `tracing` of the §8 recovery loop.
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

/// A static label for a [`RecoveryStep`] decision, for structured `tracing` of the §8 recovery loop.
pub(super) fn recovery_step_kind(step: &RecoveryStep) -> &'static str {
    match step {
        RecoveryStep::Retry { .. } => "retry",
        RecoveryStep::Rotate => "rotate",
        RecoveryStep::Compact => "compact",
        RecoveryStep::Fallback => "fallback",
        RecoveryStep::Abort => "abort",
    }
}
