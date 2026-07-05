// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The per-turn tool-call loop guardrail (§12) — a port of hermes `agent/tool_guardrails.py`.
//!
//! This is a pure, side-effect-free controller: it observes each `(name, args)` tool call in a turn
//! and returns a [`GuardrailDecision`] (allow / warn / block / halt). The turn loop in
//! [`crate::engine`] owns whether those decisions become appended result guidance, a synthetic
//! blocked result, or a controlled turn stop. One controller is created per `run_turn` call, which
//! is exactly hermes' `reset_for_turn` semantics.
//!
//! It **complements** the round-level no-progress guard ([`crate::config::Config::max_repeated_rounds`]),
//! which hashes the whole round and counts consecutive identical rounds. This guard instead tracks
//! each `(name, args)` signature across the *entire* turn (not just consecutive rounds) and escalates
//! warn→block/halt with three distinct axes: repeated identical **failure**, same-tool **failure**,
//! and idempotent **no-progress** (a read-only call returning the same result). The two compose: the
//! per-call warnings surface to the model as appended guidance while the coarse round guard still
//! ends a whole-round loop.
//!
//! **Deviation from the literal hermes port (coordinator Q4):** hermes classifies a call as
//! idempotent via two hardcoded name sets (`IDEMPOTENT_TOOL_NAMES` / `MUTATING_TOOL_NAMES`). The
//! daemon derives idempotency *structurally* instead — the caller passes `idempotent = !mutates_for(call)`
//! (see [`crate::tools::Tool::mutates_for`]), so a tool's own per-call mutation predicate is the
//! single source of truth and no name list can drift out of sync with the tool surface. Failure is
//! likewise structural: the daemon uses the tool's own [`ToolResult::ok`] flag rather than hermes'
//! `_detect_tool_failure` heuristic.

use crate::config::GuardrailConfig;
use crate::conversation::ToolResult;
use crate::repair::repair_tool_args;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A stable, non-reversible identity for a `(tool_name, canonical_args)` pair. Args are canonicalized
/// (repaired → key-sorted JSON) and hashed, so two calls with the same arguments in a different key
/// order share a signature (hermes `ToolCallSignature`). The raw args are never stored.
#[derive(Clone, PartialEq, Eq, Hash)]
struct CallSignature {
    tool_name: String,
    args_hash: u64,
}

impl CallSignature {
    fn from_call(tool_name: &str, args: &str) -> Self {
        Self {
            tool_name: tool_name.to_owned(),
            args_hash: hash_canonical_json(args),
        }
    }
}

/// What the guardrail controller decided about a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardrailAction {
    /// The call runs normally.
    Allow,
    /// The call runs, but a one-line nudge is appended to its result (the model sees the loop).
    Warn,
    /// The call must NOT run: a synthetic error result is substituted and the turn ends.
    Block,
    /// The call already ran; the turn ends after this round (too many failures of one tool).
    Halt,
}

/// A guardrail controller decision (hermes `ToolGuardrailDecision`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardrailDecision {
    /// The decided action.
    pub action: GuardrailAction,
    /// A stable machine code for the decision (for logs / synthetic results).
    pub code: &'static str,
    /// A human-readable explanation appended to guidance / synthetic results.
    pub message: String,
    /// The observed count that triggered the decision (failures or repeats).
    pub count: u32,
}

impl GuardrailDecision {
    fn allow() -> Self {
        Self {
            action: GuardrailAction::Allow,
            code: "allow",
            message: String::new(),
            count: 0,
        }
    }

    /// Whether the call is permitted to execute (allow or warn).
    pub fn allows_execution(&self) -> bool {
        matches!(self.action, GuardrailAction::Allow | GuardrailAction::Warn)
    }

    /// Whether this decision should stop the turn (block or halt).
    pub fn should_halt(&self) -> bool {
        matches!(self.action, GuardrailAction::Block | GuardrailAction::Halt)
    }
}

/// The per-turn controller for repeated failed / non-progressing tool calls (hermes
/// `ToolCallGuardrailController`). Construct one per turn; feed it [`before_call`](Self::before_call)
/// then [`after_call`](Self::after_call) for each executed call.
pub struct ToolGuardrail {
    config: GuardrailConfig,
    exact_failure: HashMap<CallSignature, u32>,
    same_tool_failure: HashMap<String, u32>,
    no_progress: HashMap<CallSignature, (u64, u32)>,
    halted: Option<GuardrailDecision>,
}

impl ToolGuardrail {
    /// A fresh controller for one turn.
    pub fn new(config: GuardrailConfig) -> Self {
        Self {
            config,
            exact_failure: HashMap::new(),
            same_tool_failure: HashMap::new(),
            no_progress: HashMap::new(),
            halted: None,
        }
    }

    /// The first decision this turn that should stop it (block/halt), if any.
    pub fn halt_decision(&self) -> Option<&GuardrailDecision> {
        self.halted.as_ref()
    }

    /// Decide whether a call should run, based on prior observations this turn (hermes `before_call`).
    /// `idempotent` is the caller's structural classification (`!tool.mutates_for(call)`). A returned
    /// [`GuardrailAction::Block`] means the call must not run; the caller substitutes
    /// [`block_result_content`].
    pub fn before_call(&mut self, name: &str, args: &str, idempotent: bool) -> GuardrailDecision {
        let signature = CallSignature::from_call(name, args);
        if !self.config.hard_stop_enabled {
            return GuardrailDecision::allow();
        }

        let exact_count = self.exact_failure.get(&signature).copied().unwrap_or(0);
        if exact_count >= self.config.exact_failure_block_after {
            let decision = GuardrailDecision {
                action: GuardrailAction::Block,
                code: "repeated_exact_failure_block",
                message: format!(
                    "Blocked {name}: the same tool call failed {exact_count} times with identical \
                     arguments. Stop retrying it unchanged; change strategy or explain the blocker."
                ),
                count: exact_count,
            };
            self.record_halt(&decision);
            return decision;
        }

        if idempotent {
            if let Some((_, repeat_count)) = self.no_progress.get(&signature).copied() {
                if repeat_count >= self.config.no_progress_block_after {
                    let decision = GuardrailDecision {
                        action: GuardrailAction::Block,
                        code: "idempotent_no_progress_block",
                        message: format!(
                            "Blocked {name}: this read-only call returned the same result \
                             {repeat_count} times. Stop repeating it unchanged; use the result \
                             already provided or try a different query."
                        ),
                        count: repeat_count,
                    };
                    self.record_halt(&decision);
                    return decision;
                }
            }
        }

        GuardrailDecision::allow()
    }

    /// Record the outcome of a call and decide whether to warn / halt (hermes `after_call`). Failure
    /// is the tool's own [`ToolResult::ok`] flag (`ok == false`). `idempotent` mirrors `before_call`.
    pub fn after_call(
        &mut self,
        name: &str,
        args: &str,
        result: &ToolResult,
        idempotent: bool,
    ) -> GuardrailDecision {
        let signature = CallSignature::from_call(name, args);
        let failed = !result.ok;

        if failed {
            let exact_count = self.exact_failure.get(&signature).copied().unwrap_or(0) + 1;
            self.exact_failure.insert(signature.clone(), exact_count);
            self.no_progress.remove(&signature);

            let same_count = self.same_tool_failure.get(name).copied().unwrap_or(0) + 1;
            self.same_tool_failure.insert(name.to_owned(), same_count);

            if self.config.hard_stop_enabled
                && same_count >= self.config.same_tool_failure_halt_after
            {
                let decision = GuardrailDecision {
                    action: GuardrailAction::Halt,
                    code: "same_tool_failure_halt",
                    message: format!(
                        "Stopped {name}: it failed {same_count} times this turn. Stop retrying the \
                         same failing tool path and choose a different approach."
                    ),
                    count: same_count,
                };
                self.record_halt(&decision);
                return decision;
            }

            if self.config.warnings_enabled && exact_count >= self.config.exact_failure_warn_after {
                return GuardrailDecision {
                    action: GuardrailAction::Warn,
                    code: "repeated_exact_failure_warning",
                    message: format!(
                        "{name} has failed {exact_count} times with identical arguments. This looks \
                         like a loop; inspect the error and change strategy instead of retrying it \
                         unchanged."
                    ),
                    count: exact_count,
                };
            }

            if self.config.warnings_enabled
                && same_count >= self.config.same_tool_failure_warn_after
            {
                return GuardrailDecision {
                    action: GuardrailAction::Warn,
                    code: "same_tool_failure_warning",
                    message: tool_failure_recovery_hint(name, same_count),
                    count: same_count,
                };
            }

            return GuardrailDecision {
                action: GuardrailAction::Allow,
                code: "allow",
                message: String::new(),
                count: exact_count,
            };
        }

        // Success: clear failure state for this signature / tool.
        self.exact_failure.remove(&signature);
        self.same_tool_failure.remove(name);

        if !idempotent {
            self.no_progress.remove(&signature);
            return GuardrailDecision::allow();
        }

        let result_hash = hash_canonical_json(&result.content);
        let repeat_count = match self.no_progress.get(&signature).copied() {
            Some((prev_hash, prev_count)) if prev_hash == result_hash => prev_count + 1,
            _ => 1,
        };
        self.no_progress
            .insert(signature, (result_hash, repeat_count));

        if self.config.warnings_enabled && repeat_count >= self.config.no_progress_warn_after {
            return GuardrailDecision {
                action: GuardrailAction::Warn,
                code: "idempotent_no_progress_warning",
                message: format!(
                    "{name} returned the same result {repeat_count} times. Use the result already \
                     provided or change the query instead of repeating it unchanged."
                ),
                count: repeat_count,
            };
        }

        GuardrailDecision {
            action: GuardrailAction::Allow,
            code: "allow",
            message: String::new(),
            count: repeat_count,
        }
    }

    fn record_halt(&mut self, decision: &GuardrailDecision) {
        if self.halted.is_none() {
            self.halted = Some(decision.clone());
        }
    }
}

/// Build the synthetic `role=tool` result content for a **blocked** call (hermes
/// `toolguard_synthetic_result`): a JSON error envelope carrying the guardrail metadata.
pub fn block_result_content(decision: &GuardrailDecision) -> String {
    serde_json::json!({
        "error": decision.message,
        "guardrail": { "code": decision.code, "count": decision.count },
    })
    .to_string()
}

/// Append runtime guidance to a tool result for a `warn`/`halt` decision (hermes
/// `append_toolguard_guidance`). No-op for allow/block or an empty message.
pub fn append_guidance(content: String, decision: &GuardrailDecision) -> String {
    if !matches!(
        decision.action,
        GuardrailAction::Warn | GuardrailAction::Halt
    ) || decision.message.is_empty()
    {
        return content;
    }
    let label = if decision.action == GuardrailAction::Halt {
        "Tool loop hard stop"
    } else {
        "Tool loop warning"
    };
    format!(
        "{content}\n\n[{label}: {code}; count={count}; {message}]",
        code = decision.code,
        count = decision.count,
        message = decision.message,
    )
}

/// Action-oriented guidance for repeated tool failures (hermes `_tool_failure_recovery_hint`).
fn tool_failure_recovery_hint(name: &str, count: u32) -> String {
    let common = format!(
        "{name} has failed {count} times this turn. This looks like a loop. Do not switch to \
         text-only replies; keep using tools, but diagnose before retrying. First inspect the \
         latest error/output and verify your assumptions. "
    );
    if name == "shell" {
        format!(
            "{common}For command failures, run a small diagnostic (e.g. `pwd && ls -la`) in the \
             same tool, then try an absolute path, a simpler command, a different working \
             directory, or a different tool such as fs read/write/edit."
        )
    } else {
        format!(
            "{common}Try different arguments, a narrower query/path, an absolute path when \
             relevant, or a different tool that can make progress. If the blocker is external, \
             report it after one diagnostic attempt instead of repeating the same failing path."
        )
    }
}

/// Hash the canonical (key-sorted, compact) JSON form of `raw` (a tool's args or a result body),
/// after the §9 repair pass. Falls back to hashing the repaired string when it is not JSON. Sorting
/// keys makes the hash order-insensitive regardless of any `serde_json` `preserve_order` feature
/// unification in the workspace (hermes uses `sort_keys=True`).
fn hash_canonical_json(raw: &str) -> u64 {
    let repaired = repair_tool_args(raw).args;
    let canonical = match serde_json::from_str::<serde_json::Value>(&repaired) {
        Ok(value) => {
            let mut out = String::new();
            canonicalize(&value, &mut out);
            out
        }
        Err(_) => repaired,
    };
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    canonical.hash(&mut hasher);
    hasher.finish()
}

/// Serialize a JSON value with object keys recursively sorted, compactly, into `out`.
fn canonicalize(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                canonicalize(&map[*key], out);
            }
            out.push('}');
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonicalize(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_hard_stop() -> GuardrailConfig {
        GuardrailConfig {
            hard_stop_enabled: true,
            ..GuardrailConfig::default()
        }
    }

    fn ok(content: &str) -> ToolResult {
        ToolResult {
            call_id: "c".into(),
            ok: true,
            content: content.into(),
        }
    }

    fn fail(content: &str) -> ToolResult {
        ToolResult {
            call_id: "c".into(),
            ok: false,
            content: content.into(),
        }
    }

    #[test]
    fn exact_failure_warns_at_two_blocks_at_five() {
        let mut g = ToolGuardrail::new(cfg_hard_stop());
        // 1st failure: allow (no warn yet).
        let d = g.after_call("shell", "{\"cmd\":\"x\"}", &fail("err"), false);
        assert_eq!(d.action, GuardrailAction::Allow);
        // 2nd identical failure: warn.
        let d = g.after_call("shell", "{\"cmd\":\"x\"}", &fail("err"), false);
        assert_eq!(d.action, GuardrailAction::Warn);
        assert_eq!(d.code, "repeated_exact_failure_warning");
        // Failures 3,4,5 keep incrementing exact_count -> 5.
        for _ in 0..3 {
            g.after_call("shell", "{\"cmd\":\"x\"}", &fail("err"), false);
        }
        // before_call now blocks (exact_count == 5 >= 5).
        let d = g.before_call("shell", "{\"cmd\":\"x\"}", false);
        assert_eq!(d.action, GuardrailAction::Block);
        assert_eq!(d.code, "repeated_exact_failure_block");
        assert!(g.halt_decision().is_some());
    }

    #[test]
    fn same_tool_failure_warns_at_three_halts_at_eight() {
        let mut g = ToolGuardrail::new(cfg_hard_stop());
        // Distinct args each time -> same-tool axis, not exact axis.
        for i in 0..2 {
            let d = g.after_call("shell", &format!("{{\"cmd\":\"a{i}\"}}"), &fail("e"), false);
            assert_eq!(d.action, GuardrailAction::Allow);
        }
        let d = g.after_call("shell", "{\"cmd\":\"a2\"}", &fail("e"), false);
        assert_eq!(d.action, GuardrailAction::Warn);
        assert_eq!(d.code, "same_tool_failure_warning");
        // Reach 8 total failures -> halt.
        for i in 3..8 {
            g.after_call("shell", &format!("{{\"cmd\":\"a{i}\"}}"), &fail("e"), false);
        }
        assert!(g.halt_decision().is_some());
        assert_eq!(g.halt_decision().unwrap().code, "same_tool_failure_halt");
    }

    #[test]
    fn idempotent_no_progress_warns_then_blocks() {
        let mut g = ToolGuardrail::new(cfg_hard_stop());
        // 1st success: allow.
        let d = g.after_call("read", "{\"path\":\"a\"}", &ok("same"), true);
        assert_eq!(d.action, GuardrailAction::Allow);
        // 2nd identical success: warn (repeat==2).
        let d = g.after_call("read", "{\"path\":\"a\"}", &ok("same"), true);
        assert_eq!(d.action, GuardrailAction::Warn);
        assert_eq!(d.code, "idempotent_no_progress_warning");
        // Reach repeat==5.
        for _ in 0..3 {
            g.after_call("read", "{\"path\":\"a\"}", &ok("same"), true);
        }
        let d = g.before_call("read", "{\"path\":\"a\"}", true);
        assert_eq!(d.action, GuardrailAction::Block);
        assert_eq!(d.code, "idempotent_no_progress_block");
    }

    #[test]
    fn changed_result_resets_no_progress_repeat() {
        let mut g = ToolGuardrail::new(cfg_hard_stop());
        g.after_call("read", "{\"path\":\"a\"}", &ok("v1"), true);
        g.after_call("read", "{\"path\":\"a\"}", &ok("v1"), true); // repeat 2
        let d = g.after_call("read", "{\"path\":\"a\"}", &ok("v2"), true); // reset -> repeat 1
        assert_eq!(d.action, GuardrailAction::Allow);
        assert_eq!(d.count, 1);
    }

    #[test]
    fn mutating_call_never_counts_as_no_progress() {
        let mut g = ToolGuardrail::new(cfg_hard_stop());
        // idempotent=false: identical successful results never warn.
        for _ in 0..6 {
            let d = g.after_call("shell", "{\"cmd\":\"echo\"}", &ok("same"), false);
            assert_eq!(d.action, GuardrailAction::Allow);
        }
        assert!(g.halt_decision().is_none());
    }

    #[test]
    fn warn_only_when_hard_stop_disabled() {
        // Default config: warnings on, hard_stop off.
        let mut g = ToolGuardrail::new(GuardrailConfig::default());
        for _ in 0..10 {
            g.after_call("read", "{\"path\":\"a\"}", &ok("same"), true);
        }
        // before_call never blocks with hard_stop disabled.
        let d = g.before_call("read", "{\"path\":\"a\"}", true);
        assert_eq!(d.action, GuardrailAction::Allow);
        assert!(g.halt_decision().is_none());
    }

    #[test]
    fn warnings_disabled_suppresses_warn() {
        let mut g = ToolGuardrail::new(GuardrailConfig {
            warnings_enabled: false,
            ..GuardrailConfig::default()
        });
        for _ in 0..4 {
            let d = g.after_call("read", "{\"path\":\"a\"}", &ok("same"), true);
            assert_eq!(d.action, GuardrailAction::Allow);
        }
    }

    #[test]
    fn canonical_args_are_key_order_insensitive() {
        assert_eq!(
            hash_canonical_json("{\"a\":1,\"b\":2}"),
            hash_canonical_json("{\"b\":2,\"a\":1}")
        );
        assert_ne!(
            hash_canonical_json("{\"a\":1}"),
            hash_canonical_json("{\"a\":2}")
        );
    }

    #[test]
    fn guidance_and_block_content_render() {
        let d = GuardrailDecision {
            action: GuardrailAction::Warn,
            code: "idempotent_no_progress_warning",
            message: "stop repeating".into(),
            count: 3,
        };
        let out = append_guidance("orig".into(), &d);
        assert!(out.starts_with("orig"));
        assert!(out.contains("Tool loop warning"));
        assert!(out.contains("count=3"));

        let block = GuardrailDecision {
            action: GuardrailAction::Block,
            code: "repeated_exact_failure_block",
            message: "blocked".into(),
            count: 5,
        };
        let body = block_result_content(&block);
        assert!(body.contains("repeated_exact_failure_block"));
        assert!(body.contains("blocked"));
        // append_guidance is a no-op for block actions.
        assert_eq!(append_guidance("x".into(), &block), "x");
    }
}
