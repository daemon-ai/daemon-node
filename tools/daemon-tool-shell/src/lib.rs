//! `daemon-tool-shell` — the command-execution tool (§12/§13), a `daemon_core::Tool`.
//!
//! Runs a transient command through the engine's [`ExecutionEnvironment`](daemon_core::ExecutionEnvironment)
//! — in the session's contained workspace, with a scrubbed child environment — never by spawning a
//! process directly. Two safety tiers gate execution (§12 preflight): a **hardline denylist** of
//! catastrophic commands is always blocked, and a **dangerous-pattern** heuristic raises a blocking
//! `HostRequest::Approval` so an operator (or the host's policy) can allow or deny it. Benign commands
//! run unattended. The result carries a structured [`ToolDetail`] (`kind = "shell"`).

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_core::{
    approve_command, Command, Effect, ExecCx, Gate, Tool, ToolCall, ToolOutcome, TurnCx,
};
use daemon_protocol::ToolDetail;
use serde::{Deserialize, Serialize};

/// Catastrophic command fragments that are *always* denied, regardless of approval policy.
const HARDLINE: &[&str] = &[
    "rm -rf /",
    "rm -fr /",
    "mkfs",
    "shutdown",
    "reboot",
    ":(){",
    "of=/dev/",
    "> /dev/sd",
];

/// The shell tool's arguments: a program plus its argument vector (no shell string-splitting, so
/// there is no shell-injection surface).
#[derive(Debug, Deserialize)]
struct ShellArgs {
    command: String,
    #[serde(default)]
    args: Vec<String>,
}

/// The structured detail attached to a shell result (opaque to the daemon; rendered by `kind`).
#[derive(Debug, Serialize)]
struct ShellDetail<'a> {
    command: &'a str,
    exit_code: i32,
    stdout_len: usize,
    stderr_len: usize,
}

/// The command-execution tool.
#[derive(Default)]
pub struct ShellTool;

impl ShellTool {
    /// A new shell tool.
    pub fn new() -> Self {
        Self
    }
}

/// Whether the command line is catastrophic and must be blocked outright.
fn is_hardline(line: &str) -> bool {
    HARDLINE.iter().any(|frag| line.contains(frag))
}

/// Whether the command is risky enough to require human/host approval before running.
fn needs_approval(program: &str, line: &str) -> bool {
    program == "sudo"
        || program == "dd"
        || line.contains("rm -rf")
        || line.contains("rm -fr")
        || line.contains("rm -r ")
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"command":{"type":"string"},"args":{"type":"array","items":{"type":"string"}}},"required":["command"]}"#
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let parsed: ShellArgs = match serde_json::from_str(&call.args) {
            Ok(args) => args,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("shell: invalid arguments: {e}"),
                )
            }
        };
        let line = if parsed.args.is_empty() {
            parsed.command.clone()
        } else {
            format!("{} {}", parsed.command, parsed.args.join(" "))
        };

        // Tier 1 — hardline: always blocked, no approval path.
        if is_hardline(&line) {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!("shell: blocked (hardline-denied command): {line}"),
            );
        }

        // Tier 2 — dangerous: gate on the session's approval policy (§12). The live host answers
        // inline; the headless/durable host parks the decision and the turn suspends.
        if needs_approval(&parsed.command, &line) {
            let prompt = format!("approve command: {line}");
            match approve_command(cx, prompt.clone()).await {
                Gate::Proceed => {}
                Gate::Reject(reason) => {
                    return ToolOutcome::text(
                        call.call_id.clone(),
                        false,
                        format!("shell: {reason}: {line}"),
                    )
                }
                Gate::Defer(job_id) => {
                    // Durable HITL: suspend awaiting the operator's decision; do not run the command.
                    return ToolOutcome::text(
                        call.call_id.clone(),
                        false,
                        format!("awaiting-approval:{job_id}"),
                    )
                    .with_effects(vec![Effect::AwaitDecision {
                        job_id,
                        call: call.clone(),
                        prompt,
                        path: None,
                    }]);
                }
            }
        }

        let cmd = Command::new(parsed.command.clone()).args(parsed.args.clone());
        let exec_cx = ExecCx { cancel: &cx.cancel };
        match cx.exec.run(cmd, &exec_cx).await {
            Ok(result) => {
                let detail = ToolDetail {
                    kind: "shell".into(),
                    body: serde_json::to_vec(&ShellDetail {
                        command: &line,
                        exit_code: result.exit_code,
                        stdout_len: result.stdout.len(),
                        stderr_len: result.stderr.len(),
                    })
                    .unwrap_or_default(),
                };
                let content = format!(
                    "exit={}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                    result.exit_code, result.stdout, result.stderr
                );
                ToolOutcome::text(call.call_id.clone(), result.ok(), content).with_detail(detail)
            }
            Err(e) => ToolOutcome::text(call.call_id.clone(), false, format!("shell: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::{Budget, SessionId};
    use daemon_core::{ApprovalPolicy, EventSink, LocalEnvironment};
    use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
    use std::path::PathBuf;
    use tokio_util::sync::CancellationToken;

    /// A host that answers every approval with a fixed decision.
    struct FixedHost(bool);
    #[async_trait]
    impl HostRequestHandler for FixedHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved(self.0),
            }
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("daemon-tool-shell-test-{tag}-{nanos}"))
    }

    async fn run(env: &LocalEnvironment, host: &dyn HostRequestHandler, args: &str) -> ToolOutcome {
        let cancel = CancellationToken::new();
        let events = EventSink::discarding();
        let cx = TurnCx {
            cancel,
            events: &events,
            host,
            session_id: SessionId::new("t"),
            budget: Budget::unlimited(),
            exec: env,
            tool_result_budget: 0,
            approval_policy: ApprovalPolicy::Ask,
            pre_approved: false,
        };
        let call = ToolCall {
            call_id: "c1".into(),
            name: "shell".into(),
            args: args.into(),
        };
        ShellTool::new().run(&call, &cx).await
    }

    #[tokio::test]
    async fn benign_command_runs_and_captures_stdout() {
        let root = temp_root("ok");
        let env = LocalEnvironment::new(&root);
        let host = FixedHost(true);
        let out = run(
            &env,
            &host,
            r#"{"command":"printf","args":["hi-%s","there"]}"#,
        )
        .await;
        assert!(
            out.result.ok,
            "benign command should succeed: {}",
            out.result.content
        );
        assert!(out.result.content.contains("hi-there"));
        assert!(out.detail.is_some());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn hardline_command_is_blocked_without_host() {
        let root = temp_root("hardline");
        let env = LocalEnvironment::new(&root);
        // A denying host proves the block is hardline (no approval is even consulted).
        let host = FixedHost(false);
        let out = run(&env, &host, r#"{"command":"rm","args":["-rf","/"]}"#).await;
        assert!(!out.result.ok);
        assert!(out.result.content.contains("hardline"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn dangerous_command_denied_by_approval() {
        let root = temp_root("deny");
        let env = LocalEnvironment::new(&root);
        let host = FixedHost(false);
        let out = run(&env, &host, r#"{"command":"sudo","args":["ls"]}"#).await;
        assert!(!out.result.ok);
        assert!(out.result.content.contains("denied"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
