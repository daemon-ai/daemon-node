// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-process` — manage background processes started with `shell(background=true)`
//! (hermes `PROCESS_SCHEMA`), a `daemon_core::Tool` over the node's resident
//! [`ProcessRegistry`](daemon_processes::ProcessRegistry).
//!
//! Actions: `list` (own processes), `poll` (status + fresh output), `log` (paged full output),
//! `wait` (block until exit / timeout / turn cancel), `kill` (SIGTERM the process group), and the
//! PTY stdin trio `write` / `submit` / `close`. Every action is scoped to the calling session's
//! own processes (ownership = the spawning `SessionId`).
//!
//! §12 per-call classification (W5 seams): the read-only actions (`list|poll|log|wait`) run
//! [`Parallel`](ToolConcurrency::Parallel) and don't checkpoint; the mutating ones
//! (`kill|write|submit|close`) are [`Exclusive`](ToolConcurrency::Exclusive) and do. `wait` opts
//! out of the engine's per-tool timeout (it self-limits, clamped to the shell default like
//! hermes).

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use daemon_core::{Tool, ToolCall, ToolConcurrency, ToolOutcome, TurnCx};
use daemon_processes::{KillResult, ProcessRegistry, ShellConfig, StdinResult, WaitStatus};
use serde::Deserialize;
use serde_json::{json, Value};

/// The process tool's arguments (hermes `PROCESS_SCHEMA`). `session_id` tolerates a number — some
/// models send the id unquoted.
#[derive(Debug, Deserialize)]
struct ProcessArgs {
    action: String,
    #[serde(default)]
    session_id: Option<Value>,
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    timeout: Option<u64>,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

/// The background-process management tool.
pub struct ProcessTool {
    procs: Arc<ProcessRegistry>,
    config: ShellConfig,
}

impl ProcessTool {
    /// A process tool over the node's resident registry, with the `[shell]` limits (the `wait`
    /// clamp).
    pub fn new(procs: Arc<ProcessRegistry>, config: ShellConfig) -> Self {
        Self { procs, config }
    }
}

/// The action named by a call's raw JSON (used by the per-call §12 classification, which must not
/// fail on malformed args — those error inside `run`).
fn action_of(call: &ToolCall) -> String {
    serde_json::from_str::<ProcessArgs>(&call.args)
        .map(|a| a.action)
        .unwrap_or_default()
}

/// Whether `action` mutates the process (kill / stdin) rather than reading it.
fn is_mutating(action: &str) -> bool {
    matches!(action, "kill" | "write" | "submit" | "close")
}

fn err_json(call_id: &str, body: Value) -> ToolOutcome {
    ToolOutcome::text(call_id.to_string(), false, body.to_string())
}

fn ok_json(call_id: &str, body: Value) -> ToolOutcome {
    ToolOutcome::text(call_id.to_string(), true, body.to_string())
}

fn not_found(call_id: &str, id: &str) -> ToolOutcome {
    err_json(
        call_id,
        json!({ "status": "not_found", "error": format!("No process with ID {id}") }),
    )
}

#[async_trait]
impl Tool for ProcessTool {
    fn name(&self) -> &str {
        "process"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"action":{"type":"string","enum":["list","poll","log","wait","kill","write","submit","close"],"description":"Action to perform on background processes: 'list' (show all), 'poll' (check status + new output), 'log' (full output with pagination), 'wait' (block until done or timeout), 'kill' (terminate), 'write' (send raw stdin data without newline), 'submit' (send data + Enter, for answering prompts), 'close' (close stdin/send EOF)."},"session_id":{"type":"string","description":"Process session ID (from shell background output). Required for all actions except 'list'."},"data":{"type":"string","description":"Text to send to process stdin (for 'write' and 'submit' actions)."},"timeout":{"type":"integer","minimum":1,"description":"Max seconds to block for 'wait'. Returns partial output on timeout."},"offset":{"type":"integer","description":"Line offset for 'log' (default: last 200 lines)."},"limit":{"type":"integer","minimum":1,"description":"Max lines to return for 'log'."}},"required":["action"]}"#
    }

    fn concurrency(&self) -> ToolConcurrency {
        // Call-independent conservative default; the per-call refinement below opens up the
        // read-only actions.
        ToolConcurrency::Exclusive
    }

    fn concurrency_for(&self, call: &ToolCall) -> ToolConcurrency {
        match action_of(call).as_str() {
            "list" | "poll" | "log" | "wait" => ToolConcurrency::Parallel,
            _ => ToolConcurrency::Exclusive,
        }
    }

    fn mutates(&self) -> bool {
        // Call-independent conservative default (kill/stdin change process state).
        true
    }

    fn mutates_for(&self, call: &ToolCall) -> bool {
        is_mutating(&action_of(call))
    }

    fn call_timeout(&self, call: &ToolCall, default: Option<Duration>) -> Option<Duration> {
        // `wait` self-limits (clamped below, hermes-style); everything else is quick.
        if action_of(call) == "wait" {
            None
        } else {
            default
        }
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let parsed: ProcessArgs = match serde_json::from_str(&call.args) {
            Ok(args) => args,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("process: invalid arguments: {e}"),
                )
            }
        };

        if parsed.action == "list" {
            // Scoped to the calling session's own processes (ownership decision: SessionId).
            let rows = self.procs.list(Some(&cx.session_id));
            return ok_json(
                &call.call_id,
                json!({ "processes": serde_json::to_value(rows).unwrap_or_default() }),
            );
        }
        if !matches!(
            parsed.action.as_str(),
            "poll" | "log" | "wait" | "kill" | "write" | "submit" | "close"
        ) {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                format!(
                    "process: unknown action: {}. Use: list, poll, log, wait, kill, write, \
                     submit, close",
                    parsed.action
                ),
            );
        }

        // Every other action targets one process; coerce a numeric id like hermes does.
        let id = match &parsed.session_id {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            _ => {
                return err_json(
                    &call.call_id,
                    json!({
                        "status": "error",
                        "error": format!("session_id is required for {}", parsed.action)
                    }),
                )
            }
        };
        // Ownership gate: a session only touches its own processes.
        match self.procs.owner_of(&id) {
            None => return not_found(&call.call_id, &id),
            Some(owner) if owner != cx.session_id => {
                return err_json(
                    &call.call_id,
                    json!({
                        "status": "forbidden",
                        "error": format!("process {id} belongs to another session")
                    }),
                )
            }
            Some(_) => {}
        }

        match parsed.action.as_str() {
            "poll" => match self.procs.poll(&id) {
                Some(result) => ok_json(
                    &call.call_id,
                    serde_json::to_value(result).unwrap_or_default(),
                ),
                None => not_found(&call.call_id, &id),
            },
            "log" => {
                let limit = parsed.limit.unwrap_or(200);
                match self.procs.read_log(&id, parsed.offset.unwrap_or(0), limit) {
                    Some(result) => ok_json(
                        &call.call_id,
                        serde_json::to_value(result).unwrap_or_default(),
                    ),
                    None => not_found(&call.call_id, &id),
                }
            }
            "wait" => {
                // Hermes clamps a wait to the configured default timeout, noting the clamp.
                let cap = self.config.timeout_default_secs;
                let (effective, note) = match parsed.timeout {
                    Some(t) if t > cap => (
                        cap,
                        Some(format!(
                            "Requested wait of {t}s was clamped to configured limit of {cap}s"
                        )),
                    ),
                    Some(t) => (t, None),
                    None => (cap, None),
                };
                match self
                    .procs
                    .wait(&id, Duration::from_secs(effective), &cx.cancel)
                    .await
                {
                    Some(result) => {
                        let mut body = serde_json::to_value(&result).unwrap_or_default();
                        match result.status {
                            WaitStatus::Timeout => {
                                body["timeout_note"] = Value::String(note.unwrap_or_else(|| {
                                    format!("Waited {effective}s, process still running")
                                }));
                            }
                            WaitStatus::Interrupted => {
                                body["note"] = Value::String(
                                    "Turn was cancelled — wait interrupted".to_string(),
                                );
                            }
                            WaitStatus::Exited => {
                                if let Some(note) = note {
                                    body["timeout_note"] = Value::String(note);
                                }
                            }
                        }
                        ok_json(&call.call_id, body)
                    }
                    None => not_found(&call.call_id, &id),
                }
            }
            "kill" => match self.procs.kill(&id, "process.kill") {
                KillResult::Killed => ok_json(
                    &call.call_id,
                    json!({
                        "status": "killed",
                        "session_id": id,
                        "completion_reason": "killed",
                        "termination_source": "process.kill",
                    }),
                ),
                KillResult::AlreadyExited { exit_code } => ok_json(
                    &call.call_id,
                    json!({ "status": "already_exited", "exit_code": exit_code }),
                ),
                KillResult::NotFound => not_found(&call.call_id, &id),
                KillResult::Failed(e) => {
                    err_json(&call.call_id, json!({ "status": "error", "error": e }))
                }
            },
            "write" | "submit" | "close" => {
                let data = parsed.data.unwrap_or_default();
                let outcome = match parsed.action.as_str() {
                    "write" => self.procs.write_stdin(&id, &data),
                    "submit" => self.procs.submit_stdin(&id, &data),
                    _ => self.procs.close_stdin(&id),
                };
                match outcome {
                    StdinResult::Ok { bytes } => {
                        let body = if parsed.action == "close" {
                            json!({ "status": "ok", "message": "EOF sent" })
                        } else {
                            json!({ "status": "ok", "bytes_written": bytes })
                        };
                        ok_json(&call.call_id, body)
                    }
                    StdinResult::NotFound => not_found(&call.call_id, &id),
                    StdinResult::AlreadyExited => err_json(
                        &call.call_id,
                        json!({ "status": "already_exited", "error": "Process has already finished" }),
                    ),
                    StdinResult::NotAvailable => err_json(
                        &call.call_id,
                        json!({
                            "status": "error",
                            "error": "Process stdin not available (piped sessions take no stdin — spawn with pty=true for interactive input)"
                        }),
                    ),
                    StdinResult::Failed(e) => {
                        err_json(&call.call_id, json!({ "status": "error", "error": e }))
                    }
                }
            }
            // Unreachable: the action set was validated above.
            _ => ToolOutcome::text(call.call_id.clone(), false, "process: unknown action"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::{Budget, SessionId};
    use daemon_core::{ApprovalPolicy, EventSink, LocalEnvironment};
    use daemon_processes::{RealClock, RegistryConfig, SpawnRequest};
    use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
    use tokio_util::sync::CancellationToken;

    struct NoopHost;
    #[async_trait]
    impl HostRequestHandler for NoopHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved(true),
            }
        }
    }

    fn registry() -> Arc<ProcessRegistry> {
        Arc::new(ProcessRegistry::new(
            RegistryConfig::default(),
            Arc::new(RealClock::new()),
        ))
    }

    fn spawn(reg: &Arc<ProcessRegistry>, owner: &str, line: &str) -> String {
        reg.spawn(SpawnRequest {
            owner: SessionId::new(owner),
            line: line.to_string(),
            cwd: std::env::temp_dir(),
            pty: false,
            notify_on_complete: false,
            watch_patterns: Vec::new(),
        })
        .expect("spawn")
        .id()
        .to_string()
    }

    async fn run_as(tool: &ProcessTool, session: &str, args: &str) -> (bool, serde_json::Value) {
        let env = LocalEnvironment::new(std::env::temp_dir().join("daemon-tool-process-test"));
        let cancel = CancellationToken::new();
        let events = EventSink::discarding();
        let host = NoopHost;
        let cx = TurnCx {
            cancel,
            events: &events,
            host: &host,
            session_id: SessionId::new(session),
            profile: None,
            budget: Budget::unlimited(),
            exec: &env,
            tool_result_budget: 0,
            approval_policy: ApprovalPolicy::Ask,
            pre_approved: false,
            checkpoints: None,
            tool_timeout: None,
        };
        let call = ToolCall {
            call_id: "c1".into(),
            name: "process".into(),
            args: args.into(),
        };
        let out = tool.run(&call, &cx).await;
        let body = serde_json::from_str(&out.result.content)
            .unwrap_or_else(|_| serde_json::Value::String(out.result.content.clone()));
        (out.result.ok, body)
    }

    #[test]
    fn per_call_classification_follows_the_action() {
        let tool = ProcessTool::new(registry(), ShellConfig::default());
        let call = |args: &str| ToolCall {
            call_id: "c".into(),
            name: "process".into(),
            args: args.into(),
        };
        for action in ["list", "poll", "log", "wait"] {
            let c = call(&format!(r#"{{"action":"{action}"}}"#));
            assert_eq!(
                tool.concurrency_for(&c),
                ToolConcurrency::Parallel,
                "{action}"
            );
            assert!(!tool.mutates_for(&c), "{action} is read-only");
        }
        for action in ["kill", "write", "submit", "close"] {
            let c = call(&format!(r#"{{"action":"{action}"}}"#));
            assert_eq!(
                tool.concurrency_for(&c),
                ToolConcurrency::Exclusive,
                "{action}"
            );
            assert!(tool.mutates_for(&c), "{action} mutates");
        }
        // `wait` opts out of the engine per-tool timeout; others keep the default.
        let default = Some(Duration::from_secs(5));
        assert_eq!(
            tool.call_timeout(&call(r#"{"action":"wait"}"#), default),
            None
        );
        assert_eq!(
            tool.call_timeout(&call(r#"{"action":"poll"}"#), default),
            default
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn list_poll_wait_kill_round_trip() {
        let reg = registry();
        let tool = ProcessTool::new(reg.clone(), ShellConfig::default());
        let id = spawn(&reg, "sess-a", "echo one; sleep 60");

        let (ok, body) = run_as(&tool, "sess-a", r#"{"action":"list"}"#).await;
        assert!(ok);
        assert_eq!(body["processes"].as_array().unwrap().len(), 1);

        let (ok, body) = run_as(
            &tool,
            "sess-a",
            &format!(r#"{{"action":"poll","session_id":"{id}"}}"#),
        )
        .await;
        assert!(ok);
        assert_eq!(body["status"], "running");

        let (ok, body) = run_as(
            &tool,
            "sess-a",
            &format!(r#"{{"action":"wait","session_id":"{id}","timeout":1}}"#),
        )
        .await;
        assert!(ok);
        assert_eq!(body["status"], "timeout");
        assert!(body["timeout_note"]
            .as_str()
            .unwrap()
            .contains("still running"));

        let (ok, body) = run_as(
            &tool,
            "sess-a",
            &format!(r#"{{"action":"kill","session_id":"{id}"}}"#),
        )
        .await;
        assert!(ok, "{body}");
        assert_eq!(body["status"], "killed");

        let (ok, body) = run_as(
            &tool,
            "sess-a",
            &format!(r#"{{"action":"poll","session_id":"{id}"}}"#),
        )
        .await;
        assert!(ok);
        assert_eq!(body["status"], "exited");
        assert_eq!(body["exit_code"], -15);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ownership_scopes_every_action_to_the_spawning_session() {
        let reg = registry();
        let tool = ProcessTool::new(reg.clone(), ShellConfig::default());
        let id = spawn(&reg, "sess-owner", "sleep 60");

        // Another session neither lists nor touches it.
        let (ok, body) = run_as(&tool, "sess-intruder", r#"{"action":"list"}"#).await;
        assert!(ok);
        assert!(body["processes"].as_array().unwrap().is_empty());
        let (ok, body) = run_as(
            &tool,
            "sess-intruder",
            &format!(r#"{{"action":"kill","session_id":"{id}"}}"#),
        )
        .await;
        assert!(!ok);
        assert_eq!(body["status"], "forbidden");

        let _ = reg.kill(&id, "test-cleanup");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_clamps_to_the_configured_default_with_a_note() {
        let reg = registry();
        let config = ShellConfig {
            timeout_default_secs: 1,
            ..ShellConfig::default()
        };
        let tool = ProcessTool::new(reg.clone(), config);
        let id = spawn(&reg, "sess-c", "sleep 60");
        let (ok, body) = run_as(
            &tool,
            "sess-c",
            &format!(r#"{{"action":"wait","session_id":"{id}","timeout":9999}}"#),
        )
        .await;
        assert!(ok);
        assert_eq!(body["status"], "timeout");
        assert!(body["timeout_note"].as_str().unwrap().contains("clamped"));
        let _ = reg.kill(&id, "test-cleanup");
    }

    #[tokio::test]
    async fn missing_session_id_and_unknown_action_error() {
        let tool = ProcessTool::new(registry(), ShellConfig::default());
        let (ok, body) = run_as(&tool, "s", r#"{"action":"poll"}"#).await;
        assert!(!ok);
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("session_id is required"));
        let (ok, body) = run_as(&tool, "s", r#"{"action":"reboot","session_id":"proc_x"}"#).await;
        assert!(!ok);
        assert!(body.as_str().unwrap_or_default().contains("unknown action"));
        // A well-formed id that is not tracked reports not_found.
        let (ok, body) = run_as(
            &tool,
            "s",
            r#"{"action":"poll","session_id":"proc_missing"}"#,
        )
        .await;
        assert!(!ok);
        assert_eq!(body["status"], "not_found");
    }
}
