// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-shell` — the command-execution tool (§12/§13), a `daemon_core::Tool`.
//!
//! **Foreground** (the default) runs a transient program+argv through the engine's
//! [`ExecutionEnvironment`](daemon_core::ExecutionEnvironment) — in the session's contained
//! workspace, with a scrubbed child environment — never by spawning a process directly, under a
//! self-managed timeout (default 180 s, hard cap 600 s → nudge to background). Oversized output is
//! truncated keeping 40% head + 60% tail (hermes `terminal_tool.py`).
//!
//! **Background** (`background=true`, optionally `pty=true`) hands the command line to the
//! host-owned [`ProcessRegistry`](daemon_processes::ProcessRegistry): the process outlives the
//! turn (a turn cancel never kills it), output rolls into a 200 KB ring, and the agent manages it
//! via the `process` tool. `notify_on_complete` injects exactly one notification turn on exit;
//! `watch_patterns` notifies on output matches under the hermes rate limits (mutually exclusive —
//! when both are set, watch is dropped in favor of notify). Background/PTY are the one approved
//! shell-string surface (`sh -c "set +m; …"`); foreground stays program+argv.
//!
//! Two safety tiers gate execution in **both** paths (§12 preflight): a **hardline denylist** of
//! catastrophic commands is always blocked, and a **dangerous-pattern** heuristic raises a blocking
//! `HostRequest::Approval` so an operator (or the host's policy) can allow or deny it. Benign
//! commands run unattended. The result carries a structured [`ToolDetail`] (`kind = "shell"`).
//!
//! Per-session working directory: `workdir` sets it for one call; a bare `cd` (`{"command":"cd",
//! "args":["dir"]}`) persists it for the session (registry-backed, in-memory). Both are contained
//! within the workspace.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use daemon_core::{
    approve_command, contain, Command, Effect, ExecCx, Gate, Tool, ToolCall, ToolOutcome, TurnCx,
};
use daemon_processes::{truncate_head_tail, ProcessRegistry, ShellConfig, SpawnRequest};
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

/// The shell tool's arguments. Foreground runs `command` + `args` as program+argv (no shell
/// string-splitting, so there is no shell-injection surface); background/PTY joins them into the
/// one approved shell line.
#[derive(Debug, Deserialize)]
struct ShellArgs {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    background: bool,
    #[serde(default)]
    timeout: Option<u64>,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default)]
    pty: bool,
    #[serde(default)]
    notify_on_complete: bool,
    #[serde(default)]
    watch_patterns: Vec<String>,
}

/// The structured detail attached to a foreground shell result (opaque to the daemon; rendered by
/// `kind`).
#[derive(Debug, Serialize)]
struct ShellDetail<'a> {
    command: &'a str,
    exit_code: i32,
    stdout_len: usize,
    stderr_len: usize,
}

/// The command-execution tool. Constructed bare ([`ShellTool::new`], foreground-only — tests,
/// minimal nodes) or with the resident process registry ([`ShellTool::with_processes`], enabling
/// background/PTY + sticky cwd).
#[derive(Default)]
pub struct ShellTool {
    procs: Option<Arc<ProcessRegistry>>,
    config: ShellConfig,
}

impl ShellTool {
    /// A foreground-only shell tool (no process registry: `background=true` errors).
    pub fn new() -> Self {
        Self::default()
    }

    /// A shell tool over the node's resident process registry (background/PTY spawn + per-session
    /// sticky cwd), with the `[shell]` limits.
    pub fn with_processes(procs: Arc<ProcessRegistry>, config: ShellConfig) -> Self {
        Self {
            procs: Some(procs),
            config,
        }
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

/// Resolve the effective working directory for this call: the explicit `workdir` (contained), else
/// the session's sticky cwd, else the workspace root. Returns the **absolute** contained path.
fn resolve_cwd(
    root: &Path,
    sticky: Option<PathBuf>,
    workdir: Option<&str>,
) -> std::io::Result<PathBuf> {
    match workdir {
        Some(dir) => contain(root, Path::new(dir)),
        None => Ok(sticky.unwrap_or_else(|| root.to_path_buf())),
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"command":{"type":"string","description":"The program to execute (foreground: program+argv; background: joined into one shell line)."},"args":{"type":"array","items":{"type":"string"},"description":"Arguments for the program."},"background":{"type":"boolean","default":false,"description":"Run in the background and return a proc_ session id immediately. Almost always pair with notify_on_complete=true — without it the process runs silently and you must remember to call process(action='poll'). Use for (1) long-lived processes that never exit (servers, watchers) and (2) long-running bounded tasks (tests, builds); prefer foreground with a generous timeout for short commands."},"timeout":{"type":"integer","minimum":1,"description":"Foreground max seconds (default 180). Returns as soon as the command finishes. Requests above the hard cap (600) are rejected — use background=true for longer work."},"workdir":{"type":"string","description":"Working directory for this call (workspace-contained). A bare cd command persists the directory for the session instead."},"pty":{"type":"boolean","default":false,"description":"Spawn the background command on a pseudo-terminal (interactive CLIs — REPLs, TUIs); enables process stdin actions (write/submit/close). Requires background=true."},"notify_on_complete":{"type":"boolean","default":false,"description":"Background only: you are notified exactly once when the process exits — the right choice for almost every long-running task. MUTUALLY EXCLUSIVE with watch_patterns (when both are set, watch_patterns is dropped)."},"watch_patterns":{"type":"array","items":{"type":"string"},"description":"Background only: substrings to watch for in output. HARD RATE LIMIT: at most 1 notification per 15s per process; after 3 consecutive rate-limited windows the watch is disabled and promoted to notify_on_complete. Use ONLY for rare one-shot signals on long-lived processes (server readiness, migration-done markers) — never for per-iteration markers in loops. MUTUALLY EXCLUSIVE with notify_on_complete."}},"required":["command"]}"#
    }

    fn mutates(&self) -> bool {
        // A shell command may write anywhere in the workspace; checkpoint before it runs.
        true
    }

    fn call_timeout(&self, _call: &ToolCall, _default: Option<Duration>) -> Option<Duration> {
        // Self-limiting: foreground applies its own configurable deadline (default 180 s, cap
        // 600 s) and background spawns return immediately — the engine's per-tool timeout stage
        // must not race it.
        None
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

        // Tier 1 — hardline: always blocked, no approval path. Applies to BOTH foreground and
        // background (the joined line is what a background shell would run).
        if is_hardline(&line) {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!("shell: blocked (hardline-denied command): {line}"),
            );
        }

        // Tier 2 — dangerous: gate on the session's approval policy (§12), again in both paths.
        // The live host answers inline; the headless/durable host parks the decision and the turn
        // suspends.
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

        let root = cx.exec.cwd().to_path_buf();
        let sticky = self
            .procs
            .as_ref()
            .filter(|_| self.config.persist_cwd)
            .and_then(|p| p.cwd_for(&cx.session_id));

        // `cd` builtin: persist the session working directory (there is no `cd` binary to exec).
        if parsed.command == "cd" && !parsed.background {
            return self.run_cd(call, cx, &root, sticky, &parsed);
        }

        let cwd = match resolve_cwd(&root, sticky, parsed.workdir.as_deref()) {
            Ok(dir) => dir,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("shell: workdir: {e}"),
                )
            }
        };

        if parsed.background {
            return self.run_background(call, cx, &line, cwd, &parsed);
        }
        if parsed.pty {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                "shell: pty=true requires background=true (interactive sessions are managed \
                 through the process tool)",
            );
        }
        self.run_foreground(call, cx, &line, &root, cwd, &parsed)
            .await
    }
}

impl ShellTool {
    /// The `cd` builtin: contain the target, verify it exists (or can be created is NOT implied —
    /// `cd` into a missing directory is an error), persist it as the session's sticky cwd.
    fn run_cd(
        &self,
        call: &ToolCall,
        cx: &TurnCx<'_>,
        root: &Path,
        sticky: Option<PathBuf>,
        parsed: &ShellArgs,
    ) -> ToolOutcome {
        let Some(procs) = &self.procs else {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                "shell: cd is unavailable (no session cwd persistence on this node)",
            );
        };
        // Resolve relative targets against the current sticky cwd (like a real shell); absolute
        // targets are contained against the workspace root.
        let target = parsed.args.first().map(String::as_str).unwrap_or(".");
        let base = sticky.unwrap_or_else(|| root.to_path_buf());
        let requested = if Path::new(target).is_absolute() {
            PathBuf::from(target)
        } else {
            base.join(target)
        };
        let resolved = match contain(root, &requested) {
            Ok(dir) => dir,
            Err(e) => {
                return ToolOutcome::text(call.call_id.clone(), false, format!("shell: cd: {e}"))
            }
        };
        if !resolved.is_dir() {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!("shell: cd: no such directory: {}", resolved.display()),
            );
        }
        procs.set_cwd(&cx.session_id, resolved.clone());
        ToolOutcome::text(
            call.call_id.clone(),
            true,
            format!("cwd is now {}", resolved.display()),
        )
    }

    /// Spawn a tracked background process through the registry and return its session id.
    fn run_background(
        &self,
        call: &ToolCall,
        cx: &TurnCx<'_>,
        line: &str,
        cwd: PathBuf,
        parsed: &ShellArgs,
    ) -> ToolOutcome {
        let Some(procs) = &self.procs else {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                "shell: background=true is unavailable (no process registry on this node)",
            );
        };
        // Mutual exclusion (hermes `_resolve_notification_flag_conflict`): notify_on_complete is
        // the more useful signal; when both are set, watch_patterns is dropped with a note.
        let mut watch_patterns = parsed.watch_patterns.clone();
        let mut conflict_note = None;
        if parsed.notify_on_complete && !watch_patterns.is_empty() {
            watch_patterns.clear();
            conflict_note = Some(
                "watch_patterns ignored because notify_on_complete=true; the two are mutually \
                 exclusive",
            );
        }
        let session = match procs.spawn(SpawnRequest {
            owner: cx.session_id.clone(),
            line: line.to_string(),
            cwd,
            pty: parsed.pty,
            notify_on_complete: parsed.notify_on_complete,
            watch_patterns: watch_patterns.clone(),
        }) {
            Ok(session) => session,
            Err(e) => return ToolOutcome::text(call.call_id.clone(), false, format!("shell: {e}")),
        };
        let mut result = serde_json::json!({
            "status": "running",
            "session_id": session.id(),
            "pid": session.pid(),
            "command": line,
        });
        if parsed.notify_on_complete {
            result["notify_on_complete"] = serde_json::Value::Bool(true);
        }
        if !watch_patterns.is_empty() {
            result["watch_patterns"] = serde_json::json!(watch_patterns);
        }
        if let Some(note) = conflict_note {
            result["watch_patterns_ignored"] = serde_json::Value::String(note.to_string());
        }
        // Nudge (hermes parity): background without notify/watch runs silently — the agent has no
        // way to learn it finished short of polling.
        if !parsed.notify_on_complete && watch_patterns.is_empty() {
            result["nudge"] = serde_json::Value::String(
                "background=true without notify_on_complete=true runs SILENTLY: you will not \
                 learn when it finishes unless you call process(action='poll'). For bounded \
                 tasks set notify_on_complete=true."
                    .to_string(),
            );
        }
        let detail = ToolDetail {
            kind: "shell".into(),
            body: serde_json::to_vec(&result).unwrap_or_default(),
        };
        ToolOutcome::text(call.call_id.clone(), true, result.to_string()).with_detail(detail)
    }

    /// Run a transient foreground command under the self-managed timeout, truncating oversized
    /// output 40% head / 60% tail.
    async fn run_foreground(
        &self,
        call: &ToolCall,
        cx: &TurnCx<'_>,
        line: &str,
        root: &Path,
        cwd: PathBuf,
        parsed: &ShellArgs,
    ) -> ToolOutcome {
        // Timeout policy (hermes): default when unset; a request above the hard cap is REJECTED
        // with a background nudge rather than clamped.
        let max = self.config.timeout_max_secs;
        let effective = match parsed.timeout {
            Some(t) if t > max => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!(
                        "shell: foreground timeout {t}s exceeds the maximum {max}s. Use \
                         background=true with notify_on_complete=true for long-running commands."
                    ),
                )
            }
            Some(t) => t,
            None => self.config.timeout_default_secs,
        };

        let mut cmd = Command::new(parsed.command.clone()).args(parsed.args.clone());
        // Only pass a cwd when it differs from the root (byte-identical legacy behavior otherwise).
        if cwd != *root {
            cmd = cmd.cwd(cwd);
        }

        // Self-managed deadline: a child token the timer cancels — the environment kills the
        // process on cancellation, so a timed-out command never lingers. A parent (turn) cancel
        // propagates through the same token.
        let timeout_token = cx.cancel.child_token();
        let timer_token = timeout_token.clone();
        let timer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(effective)).await;
            timer_token.cancel();
        });
        let exec_cx = ExecCx {
            cancel: &timeout_token,
        };
        let outcome = cx.exec.run(cmd, &exec_cx).await;
        timer.abort();
        let timed_out = timeout_token.is_cancelled() && !cx.cancel.is_cancelled();

        match outcome {
            Ok(result) => {
                let detail = ToolDetail {
                    kind: "shell".into(),
                    body: serde_json::to_vec(&ShellDetail {
                        command: line,
                        exit_code: result.exit_code,
                        stdout_len: result.stdout.len(),
                        stderr_len: result.stderr.len(),
                    })
                    .unwrap_or_default(),
                };
                let stdout = truncate_head_tail(
                    &result.stdout,
                    self.config.truncate_max_bytes,
                    self.config.truncate_head_pct,
                );
                let stderr = truncate_head_tail(
                    &result.stderr,
                    self.config.truncate_max_bytes,
                    self.config.truncate_head_pct,
                );
                let timeout_note = if timed_out {
                    format!(
                        "\n[timed out after {effective}s — the process was killed. Use \
                         background=true with notify_on_complete=true for long-running commands.]"
                    )
                } else {
                    String::new()
                };
                let content = format!(
                    "exit={}{}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                    result.exit_code, timeout_note, stdout, stderr
                );
                ToolOutcome::text(call.call_id.clone(), result.ok() && !timed_out, content)
                    .with_detail(detail)
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
    use daemon_processes::{RealClock, RegistryConfig};
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

    fn registry() -> Arc<ProcessRegistry> {
        Arc::new(ProcessRegistry::new(
            RegistryConfig::default(),
            Arc::new(RealClock::new()),
        ))
    }

    async fn run_tool(
        tool: &ShellTool,
        env: &LocalEnvironment,
        host: &dyn HostRequestHandler,
        args: &str,
    ) -> ToolOutcome {
        let cancel = CancellationToken::new();
        let events = EventSink::discarding();
        let cx = TurnCx {
            cancel,
            events: &events,
            host,
            session_id: SessionId::new("t"),
            profile: None,
            budget: Budget::unlimited(),
            exec: env,
            tool_result_budget: 0,
            approval_policy: ApprovalPolicy::Ask,
            pre_approved: false,
            checkpoints: None,
            tool_timeout: None,
        };
        let call = ToolCall {
            call_id: "c1".into(),
            name: "shell".into(),
            args: args.into(),
        };
        tool.run(&call, &cx).await
    }

    async fn run(env: &LocalEnvironment, host: &dyn HostRequestHandler, args: &str) -> ToolOutcome {
        run_tool(&ShellTool::new(), env, host, args).await
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

    /// The safety tiers gate the background path too: a hardline line never spawns, and a
    /// dangerous line is denied by the approval host before the registry sees it.
    #[tokio::test]
    async fn background_path_keeps_denylist_and_approval() {
        let root = temp_root("bg-deny");
        let env = LocalEnvironment::new(&root);
        let tool = ShellTool::with_processes(registry(), ShellConfig::default());
        let out = run_tool(
            &tool,
            &env,
            &FixedHost(true),
            r#"{"command":"rm","args":["-rf","/"],"background":true}"#,
        )
        .await;
        assert!(!out.result.ok);
        assert!(out.result.content.contains("hardline"));

        let out = run_tool(
            &tool,
            &env,
            &FixedHost(false),
            r#"{"command":"sudo","args":["ls"],"background":true}"#,
        )
        .await;
        assert!(!out.result.ok);
        assert!(out.result.content.contains("denied"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn background_spawn_returns_a_proc_session_id() {
        let root = temp_root("bg");
        let env = LocalEnvironment::new(&root);
        let procs = registry();
        let tool = ShellTool::with_processes(procs.clone(), ShellConfig::default());
        let out = run_tool(
            &tool,
            &env,
            &FixedHost(true),
            r#"{"command":"echo","args":["bg-marker"],"background":true}"#,
        )
        .await;
        assert!(out.result.ok, "{}", out.result.content);
        let body: serde_json::Value = serde_json::from_str(&out.result.content).unwrap();
        assert_eq!(body["status"], "running");
        let id = body["session_id"].as_str().unwrap();
        assert!(id.starts_with("proc_"));
        // Silent-background nudge present when neither notify nor watch is set.
        assert!(body["nudge"].as_str().unwrap().contains("SILENTLY"));
        let result = procs
            .wait(id, Duration::from_secs(10), &CancellationToken::new())
            .await
            .expect("tracked");
        assert!(result.output.contains("bg-marker"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn notify_and_watch_are_mutually_exclusive() {
        let root = temp_root("bg-conflict");
        let env = LocalEnvironment::new(&root);
        let tool = ShellTool::with_processes(registry(), ShellConfig::default());
        let out = run_tool(
            &tool,
            &env,
            &FixedHost(true),
            r#"{"command":"sleep","args":["60"],"background":true,"notify_on_complete":true,"watch_patterns":["READY"]}"#,
        )
        .await;
        assert!(out.result.ok, "{}", out.result.content);
        let body: serde_json::Value = serde_json::from_str(&out.result.content).unwrap();
        assert_eq!(body["notify_on_complete"], true);
        assert!(body.get("watch_patterns").is_none(), "watch dropped");
        assert!(body["watch_patterns_ignored"]
            .as_str()
            .unwrap()
            .contains("mutually exclusive"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn foreground_timeout_over_the_cap_is_rejected_with_a_nudge() {
        let root = temp_root("cap");
        let env = LocalEnvironment::new(&root);
        let out = run(
            &env,
            &FixedHost(true),
            r#"{"command":"sleep","args":["1"],"timeout":601}"#,
        )
        .await;
        assert!(!out.result.ok);
        assert!(out.result.content.contains("maximum 600s"));
        assert!(out.result.content.contains("background=true"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn foreground_timeout_kills_and_reports() {
        let root = temp_root("timeout");
        let env = LocalEnvironment::new(&root);
        let out = run(
            &env,
            &FixedHost(true),
            r#"{"command":"sleep","args":["300"],"timeout":1}"#,
        )
        .await;
        assert!(!out.result.ok);
        assert!(
            out.result.content.contains("timed out after 1s"),
            "{}",
            out.result.content
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn workdir_runs_in_subdir_and_escape_is_rejected() {
        let root = temp_root("workdir");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        let env = LocalEnvironment::new(&root);
        let out = run(
            &env,
            &FixedHost(true),
            r#"{"command":"pwd","workdir":"sub"}"#,
        )
        .await;
        assert!(out.result.ok);
        assert!(out.result.content.contains("sub"));

        let out = run(
            &env,
            &FixedHost(true),
            r#"{"command":"pwd","workdir":"../outside"}"#,
        )
        .await;
        assert!(!out.result.ok);
        assert!(out.result.content.contains("workdir"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn cd_persists_the_session_cwd_across_calls() {
        let root = temp_root("cd");
        std::fs::create_dir_all(root.join("proj/src")).unwrap();
        let env = LocalEnvironment::new(&root);
        let tool = ShellTool::with_processes(registry(), ShellConfig::default());
        let host = FixedHost(true);

        let out = run_tool(&tool, &env, &host, r#"{"command":"cd","args":["proj"]}"#).await;
        assert!(out.result.ok, "{}", out.result.content);
        // The next call runs in the sticky cwd; a relative cd resolves against it.
        let out = run_tool(&tool, &env, &host, r#"{"command":"pwd"}"#).await;
        assert!(
            out.result.content.contains("proj"),
            "{}",
            out.result.content
        );
        let out = run_tool(&tool, &env, &host, r#"{"command":"cd","args":["src"]}"#).await;
        assert!(out.result.ok, "{}", out.result.content);
        assert!(out.result.content.ends_with("proj/src"));
        // Escaping cd is rejected; a missing directory is an error.
        let out = run_tool(
            &tool,
            &env,
            &host,
            r#"{"command":"cd","args":["../../.."]}"#,
        )
        .await;
        assert!(!out.result.ok);
        let out = run_tool(&tool, &env, &host, r#"{"command":"cd","args":["nope"]}"#).await;
        assert!(!out.result.ok);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn background_without_registry_errors_clearly() {
        let root = temp_root("noreg");
        let env = LocalEnvironment::new(&root);
        let out = run(
            &env,
            &FixedHost(true),
            r#"{"command":"echo","args":["x"],"background":true}"#,
        )
        .await;
        assert!(!out.result.ok);
        assert!(out.result.content.contains("no process registry"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn pty_requires_background() {
        let root = temp_root("ptyfg");
        let env = LocalEnvironment::new(&root);
        let out = run(
            &env,
            &FixedHost(true),
            r#"{"command":"python3","pty":true}"#,
        )
        .await;
        assert!(!out.result.ok);
        assert!(out.result.content.contains("requires background=true"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
