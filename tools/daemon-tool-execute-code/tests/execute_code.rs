// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Integration tests for the `execute_code` tool.
//!
//! Tests that need to actually run Python skip cleanly (early-return with a note) when no usable
//! `python3` is on `PATH`; the bwrap-containment test additionally infers bubblewrap usability from
//! the tool's own setup-error and skips when the sandbox is unavailable (user namespaces off / bwrap
//! absent), so the suite is green on CI hosts without either.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use daemon_common::{Budget, JobId, SessionId};
use daemon_core::{
    ApprovalPolicy, Effect, EventSink, LocalEnvironment, Tool, ToolCall, ToolOutcome, TurnCx,
};
use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
use daemon_tool_execute_code::{
    ExecuteCodeSettings, ExecuteCodeTool, Mode, NetworkPolicy, SandboxPolicy,
};
use tokio_util::sync::CancellationToken;

/// A host that approves everything (used when the policy already decides, so it is never consulted).
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

/// A host that parks every approval durably (the headless HITL path → `Gate::Defer`).
struct DeferHost;
#[async_trait]
impl HostRequestHandler for DeferHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Deferred(JobId::new("job-defer")),
        }
    }
}

fn temp_root(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("daemon-tool-exec-{tag}-{nanos}"))
}

fn settings(sandbox: SandboxPolicy) -> ExecuteCodeSettings {
    ExecuteCodeSettings {
        default_mode: Mode::Project,
        timeout: Duration::from_secs(30),
        max_stdout_bytes: 50_000,
        max_stderr_bytes: 10_000,
        sandbox,
        network: NetworkPolicy::Off,
    }
}

/// JSON args from a code string (escaping handled by serde).
fn args(code: &str) -> String {
    serde_json::json!({ "code": code }).to_string()
}

/// JSON args with an explicit mode override.
fn args_mode(code: &str, mode: &str) -> String {
    serde_json::json!({ "code": code, "mode": mode }).to_string()
}

/// Whether a usable Python (>= 3.8) is on PATH; tests that run code skip when false.
fn python_available() -> bool {
    ["python3", "python"].iter().any(|p| {
        std::process::Command::new(p)
            .arg("-c")
            .arg("import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Run the tool once against a fresh workspace rooted at `root` with the given policy + cancel token.
async fn run_in(
    root: &Path,
    settings: ExecuteCodeSettings,
    policy: ApprovalPolicy,
    host: &dyn HostRequestHandler,
    cancel: CancellationToken,
    args: &str,
) -> ToolOutcome {
    let env = LocalEnvironment::new(root);
    let events = EventSink::discarding();
    let cx = TurnCx {
        cancel,
        events: &events,
        host,
        session_id: SessionId::new("t"),
        profile: None,
        budget: Budget::unlimited(),
        exec: &env,
        tool_result_budget: 0,
        approval_policy: policy,
        pre_approved: false,
        checkpoints: None,
        tool_timeout: None,
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: "execute_code".into(),
        args: args.into(),
    };
    ExecuteCodeTool::new(settings).run(&call, &cx).await
}

/// Convenience: the common auto-allow, plain-sandbox run.
async fn run_plain(root: &Path, args: &str) -> ToolOutcome {
    run_in(
        root,
        settings(SandboxPolicy::Plain),
        ApprovalPolicy::AutoAllow,
        &NoopHost,
        CancellationToken::new(),
        args,
    )
    .await
}

/// Parse the `sandboxed` flag out of the structured detail body.
fn detail_sandboxed(out: &ToolOutcome) -> Option<bool> {
    let detail = out.detail.as_ref()?;
    let v: serde_json::Value = serde_json::from_slice(&detail.body).ok()?;
    v.get("sandboxed").and_then(|b| b.as_bool())
}

/// Parse the `backend` label (`bwrap`/`landlock`/`sandbox-exec`/`plain`) out of the detail body.
fn detail_backend(out: &ToolOutcome) -> Option<String> {
    let detail = out.detail.as_ref()?;
    let v: serde_json::Value = serde_json::from_slice(&detail.body).ok()?;
    v.get("backend")
        .and_then(|b| b.as_str())
        .map(str::to_string)
}

// --- Test 1: argument parsing (no python needed) ---------------------------------------------

#[tokio::test]
async fn invalid_json_and_empty_code_fail_cleanly() {
    let root = temp_root("args");
    let bad = run_plain(&root, "this is not json").await;
    assert!(!bad.result.ok);
    assert!(bad.result.content.contains("invalid arguments"));

    let empty = run_plain(&root, &args("   \n  ")).await;
    assert!(!empty.result.ok);
    assert!(empty.result.content.contains("no code provided"));
    let _ = std::fs::remove_dir_all(&root);
}

// --- Test 9: approval gate (no python needed — decided before spawn) --------------------------

#[tokio::test]
async fn deny_policy_refuses_without_running() {
    let root = temp_root("deny");
    let out = run_in(
        &root,
        settings(SandboxPolicy::Plain),
        ApprovalPolicy::Deny,
        &NoopHost,
        CancellationToken::new(),
        &args("print('should not run')"),
    )
    .await;
    assert!(!out.result.ok);
    assert!(out.result.content.contains("denied"));
    // Nothing was staged/executed.
    assert!(!root.join(".execute_code").exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn ask_policy_defers_durably_with_await_effect() {
    let root = temp_root("defer");
    let out = run_in(
        &root,
        settings(SandboxPolicy::Plain),
        ApprovalPolicy::Ask,
        &DeferHost,
        CancellationToken::new(),
        &args("print('later')"),
    )
    .await;
    assert!(!out.result.ok);
    assert!(out.result.content.starts_with("awaiting-approval:"));
    assert_eq!(out.effects.len(), 1);
    assert!(matches!(out.effects[0], Effect::AwaitDecision { .. }));
    let _ = std::fs::remove_dir_all(&root);
}

// --- Test 2: success --------------------------------------------------------------------------

#[tokio::test]
async fn prints_to_stdout_on_success() {
    if !python_available() {
        eprintln!("skipping prints_to_stdout_on_success: no usable python3 on PATH");
        return;
    }
    let root = temp_root("ok");
    let out = run_plain(&root, &args("print('hello-exec')")).await;
    assert!(out.result.ok, "expected success: {}", out.result.content);
    assert!(out.result.content.contains("\"status\":\"success\""));
    assert!(out.result.content.contains("hello-exec"));
    assert!(out.result.content.contains("\"duration_seconds\""));
    assert!(out.result.content.contains("\"tool_calls_made\":0"));
    assert!(out.detail.is_some());
    let _ = std::fs::remove_dir_all(&root);
}

// --- Test 3: non-zero exit + stderr -----------------------------------------------------------

#[tokio::test]
async fn nonzero_exit_reports_error_and_stderr() {
    if !python_available() {
        eprintln!("skipping nonzero_exit_reports_error_and_stderr: no usable python3 on PATH");
        return;
    }
    let root = temp_root("err");
    let out = run_plain(
        &root,
        &args("import sys; sys.stderr.write('boom\\n'); sys.exit(2)"),
    )
    .await;
    assert!(!out.result.ok);
    assert!(out.result.content.contains("\"status\":\"error\""));
    assert!(out.result.content.contains("boom"));
    let _ = std::fs::remove_dir_all(&root);
}

// --- Test 4: stdout cap + truncation notice ---------------------------------------------------

#[tokio::test]
async fn large_stdout_is_truncated_head_tail() {
    if !python_available() {
        eprintln!("skipping large_stdout_is_truncated_head_tail: no usable python3 on PATH");
        return;
    }
    let root = temp_root("cap");
    let mut s = settings(SandboxPolicy::Plain);
    s.max_stdout_bytes = 200;
    let out = run_in(
        &root,
        s,
        ApprovalPolicy::AutoAllow,
        &NoopHost,
        CancellationToken::new(),
        &args("print('A'*100); print('Z'*2000)"),
    )
    .await;
    assert!(out.result.ok, "expected success: {}", out.result.content);
    assert!(out.result.content.contains("OUTPUT TRUNCATED"));
    // Head (leading A's) and tail (trailing Z's) both survive.
    assert!(out.result.content.contains("AAAA"));
    assert!(out.result.content.contains("ZZZZ"));
    let _ = std::fs::remove_dir_all(&root);
}

// --- Test 5: timeout --------------------------------------------------------------------------

#[tokio::test]
async fn timeout_kills_and_reports() {
    if !python_available() {
        eprintln!("skipping timeout_kills_and_reports: no usable python3 on PATH");
        return;
    }
    let root = temp_root("timeout");
    let mut s = settings(SandboxPolicy::Plain);
    s.timeout = Duration::from_secs(1);
    let out = run_in(
        &root,
        s,
        ApprovalPolicy::AutoAllow,
        &NoopHost,
        CancellationToken::new(),
        &args("import time; time.sleep(60)"),
    )
    .await;
    assert!(!out.result.ok);
    assert!(out.result.content.contains("\"status\":\"timeout\""));
    assert!(out.result.content.contains("timed out"));
    let _ = std::fs::remove_dir_all(&root);
}

// --- Test 6: cooperative cancel ---------------------------------------------------------------

#[tokio::test]
async fn cancel_interrupts_running_script() {
    if !python_available() {
        eprintln!("skipping cancel_interrupts_running_script: no usable python3 on PATH");
        return;
    }
    let root = temp_root("cancel");
    let env = LocalEnvironment::new(&root);
    let events = EventSink::discarding();
    let host = NoopHost;
    let token = CancellationToken::new();
    let cx = TurnCx {
        cancel: token.clone(),
        events: &events,
        host: &host,
        session_id: SessionId::new("t"),
        profile: None,
        budget: Budget::unlimited(),
        exec: &env,
        tool_result_budget: 0,
        approval_policy: ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
        tool_timeout: None,
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: "execute_code".into(),
        args: args("import time; time.sleep(60)"),
    };
    let tool = ExecuteCodeTool::new(settings(SandboxPolicy::Plain));
    let run_fut = tool.run(&call, &cx);
    let cancel_fut = async {
        tokio::time::sleep(Duration::from_millis(400)).await;
        token.cancel();
    };
    let (out, ()) = tokio::join!(run_fut, cancel_fut);
    assert!(!out.result.ok);
    assert!(out.result.content.contains("\"status\":\"interrupted\""));
    let _ = std::fs::remove_dir_all(&root);
}

// --- Test 8: project vs strict CWD ------------------------------------------------------------

#[tokio::test]
async fn project_and_strict_modes_use_expected_cwd() {
    if !python_available() {
        eprintln!("skipping project_and_strict_modes_use_expected_cwd: no usable python3 on PATH");
        return;
    }
    let root = temp_root("cwd");
    // Project mode: CWD is the workspace root (never the staging dir).
    let proj = run_plain(&root, &args("import os; print(os.getcwd())")).await;
    assert!(proj.result.ok, "project: {}", proj.result.content);
    assert!(!proj.result.content.contains(".execute_code"));

    // Strict mode: CWD is the isolated staging dir under the workspace.
    let strict = run_in(
        &root,
        settings(SandboxPolicy::Plain),
        ApprovalPolicy::AutoAllow,
        &NoopHost,
        CancellationToken::new(),
        &args_mode("import os; print(os.getcwd())", "strict"),
    )
    .await;
    assert!(strict.result.ok, "strict: {}", strict.result.content);
    assert!(strict.result.content.contains(".execute_code"));
    // The override is reflected in the detail envelope.
    let detail: serde_json::Value =
        serde_json::from_slice(&strict.detail.expect("detail").body).unwrap();
    assert_eq!(detail.get("mode").and_then(|m| m.as_str()), Some("strict"));
    let _ = std::fs::remove_dir_all(&root);
}

// --- Test 7: containment (plain CWD correctness + bwrap escape block) -------------------------

#[tokio::test]
async fn plain_relative_write_lands_in_workspace() {
    if !python_available() {
        eprintln!("skipping plain_relative_write_lands_in_workspace: no usable python3 on PATH");
        return;
    }
    let root = temp_root("plain-write");
    let out = run_plain(&root, &args("open('made-in-ws.txt', 'w').write('hi')")).await;
    assert!(out.result.ok, "expected success: {}", out.result.content);
    // Project-mode CWD is the workspace root, so the relative write lands inside it.
    assert!(root.join("made-in-ws.txt").exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn bwrap_blocks_out_of_workspace_write() {
    if !python_available() {
        eprintln!("skipping bwrap_blocks_out_of_workspace_write: no usable python3 on PATH");
        return;
    }
    // Nest the workspace so `../escape.txt` targets a private per-test dir, not the shared tmp root.
    let base = temp_root("bwrap-escape");
    let ws = base.join("ws");
    std::fs::create_dir_all(&ws).expect("create ws");
    let escape = base.join("escape.txt");

    let out = run_in(
        &ws,
        settings(SandboxPolicy::Require),
        ApprovalPolicy::AutoAllow,
        &NoopHost,
        CancellationToken::new(),
        &args(&format!(
            "open({:?}, 'w').write('escaped')",
            "../escape.txt"
        )),
    )
    .await;

    if out.result.content.contains("unavailable") {
        eprintln!(
            "skipping bwrap_blocks_out_of_workspace_write: bubblewrap not usable on this host"
        );
        let _ = std::fs::remove_dir_all(&base);
        return;
    }
    // The security invariant: whatever the script does inside its namespace, the write must never
    // reach the real out-of-workspace path on the host. (bwrap binds only the workspace RW; the
    // parent is either unmounted or a private tmpfs, so the escape stays inside the sandbox.)
    assert_eq!(detail_sandboxed(&out), Some(true), "run must be sandboxed");
    assert!(
        !escape.exists(),
        "escape write must not reach the host filesystem"
    );
    let _ = std::fs::remove_dir_all(&base);
}

// --- Test 10: sandbox selection ---------------------------------------------------------------

#[tokio::test]
async fn none_policy_runs_unsandboxed() {
    if !python_available() {
        eprintln!("skipping none_policy_runs_unsandboxed: no usable python3 on PATH");
        return;
    }
    let root = temp_root("nosandbox");
    let out = run_plain(&root, &args("print('plain')")).await;
    assert!(out.result.ok, "expected success: {}", out.result.content);
    assert_eq!(detail_sandboxed(&out), Some(false));
    // `Plain` is the explicit unconfined backend.
    assert_eq!(detail_backend(&out).as_deref(), Some("plain"));
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn required_bwrap_either_sandboxes_or_reports_unavailable() {
    if !python_available() {
        eprintln!(
            "skipping required_bwrap_either_sandboxes_or_reports_unavailable: no usable python3"
        );
        return;
    }
    let root = temp_root("reqbwrap");
    let out = run_in(
        &root,
        settings(SandboxPolicy::Require),
        ApprovalPolicy::AutoAllow,
        &NoopHost,
        CancellationToken::new(),
        &args("print('probe')"),
    )
    .await;
    if out.result.ok {
        // A kernel backend was usable → the run was actually confined (bwrap on this host, or the
        // Landlock+seccomp fallback where userns is off).
        assert_eq!(detail_sandboxed(&out), Some(true));
        assert!(
            matches!(
                detail_backend(&out).as_deref(),
                Some("bwrap") | Some("landlock")
            ),
            "Require must resolve to a kernel backend, got {:?}",
            detail_backend(&out)
        );
        assert!(out.result.content.contains("probe"));
    } else {
        // No backend usable → a clear setup error, never a silent unsandboxed run (fail closed).
        assert!(out.result.content.contains("unavailable"));
    }
    let _ = std::fs::remove_dir_all(&root);
}
