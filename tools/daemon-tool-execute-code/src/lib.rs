// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-execute-code` — the `execute_code` tool (§12/§13), a `daemon_core::Tool`.
//!
//! Runs a one-shot Python script (no persistent kernel) in the session's contained workspace and
//! returns only its stdout, mirroring hermes' `execute_code` semantics. Two modes: `project` (the
//! venv-aware interpreter + the workspace as CWD, so project deps and files resolve) and `strict`
//! (the system interpreter + an isolated staging dir under the workspace). Execution is bounded by a
//! self-managed timeout (so it opts out of the engine's per-tool timeout stage), output is capped
//! (50 KB stdout head/tail, 10 KB stderr head), and cancellation kills the whole process group.
//!
//! On Linux, when [bubblewrap](https://github.com/containers/bubblewrap) is available *and usable*
//! (user namespaces enabled), the script runs inside a bwrap sandbox (read-only `/nix/store` + system
//! paths, a read-write bind of only the CWD, a private `/tmp`, and — by default — no network). When
//! bwrap is absent or unusable the tool falls back to a plain subprocess; in that mode the OS jail is
//! gone but the tool's own file staging + CWD stay workspace-contained and arbitrary code still runs
//! under the daemon's uid, exactly as hermes' unsandboxed path does.
//!
//! Arbitrary code is gated by the same [`ApprovalPolicy`](daemon_core::ApprovalPolicy) as a dangerous
//! shell command (fleet `AutoAllow`; interactive `Ask` — inline on the live host, durable defer on the
//! headless host), and the tool declares [`mutates`](daemon_core::Tool::mutates) so the pipeline
//! checkpoints before it runs.

#![forbid(unsafe_code)]

mod exec;
mod python;
mod sandbox;

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use daemon_core::{
    approve_command, ContainedRoot, Effect, Gate, Tool, ToolCall, ToolOutcome, TurnCx,
};
use daemon_protocol::ToolDetail;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use exec::{OutputCaps, RunOutcome, Status};

/// The default self-managed wall-clock timeout (hermes `DEFAULT_TIMEOUT`, 5 minutes).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
/// The default stdout byte cap (hermes `MAX_STDOUT_BYTES`, 50 KB), split 40 % head / 60 % tail.
pub const DEFAULT_MAX_STDOUT_BYTES: usize = 50_000;
/// The default stderr byte cap (hermes `MAX_STDERR_BYTES`, 10 KB), head-only.
pub const DEFAULT_MAX_STDERR_BYTES: usize = 10_000;

/// The execute_code working-directory / interpreter mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Run in the session's workspace root with the venv-aware interpreter (project deps + files
    /// resolve, like the shell tool). The default.
    #[default]
    Project,
    /// Run in an isolated staging dir under the workspace with the system interpreter (reproducible;
    /// project deps + relative paths do not resolve).
    Strict,
}

impl Mode {
    /// The lowercase wire label (`"project"` / `"strict"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Project => "project",
            Mode::Strict => "strict",
        }
    }
}

/// The OS-sandbox posture for the child process. Selects among the per-platform kernel backends
/// (Linux bwrap → Landlock+seccomp; macOS `sandbox-exec`) behind one policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SandboxPolicy {
    /// Use the strongest available kernel backend for this platform; if none is usable, run a plain
    /// (unconfined) subprocess after logging a warning. The default.
    #[default]
    Auto,
    /// Require a kernel backend: fail the call if none is usable (no silent unsandboxed run). The
    /// legacy `bwrap` label still parses to this posture.
    #[serde(alias = "bwrap")]
    Require,
    /// Never sandbox — an explicit, high-friction operator choice for an unconfined subprocess. The
    /// legacy `none` label still parses to this posture.
    #[serde(alias = "none")]
    Plain,
}

/// The child's network policy (only enforced under the bwrap sandbox).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetworkPolicy {
    /// No network namespace in the sandbox (`--unshare-net`). The default.
    #[default]
    Off,
    /// Share the host network (no `--unshare-net`). Needed for scripts that fetch / pip-install.
    Shared,
}

/// Resolved settings for [`ExecuteCodeTool`], built by the host from `[execute_code]` config.
#[derive(Clone, Copy, Debug)]
pub struct ExecuteCodeSettings {
    /// The default mode when a call does not override it.
    pub default_mode: Mode,
    /// The self-managed wall-clock timeout.
    pub timeout: Duration,
    /// The stdout byte cap (head/tail).
    pub max_stdout_bytes: usize,
    /// The stderr byte cap (head-only).
    pub max_stderr_bytes: usize,
    /// The OS-sandbox policy.
    pub sandbox: SandboxPolicy,
    /// The child network policy.
    pub network: NetworkPolicy,
}

impl Default for ExecuteCodeSettings {
    fn default() -> Self {
        Self {
            default_mode: Mode::Project,
            timeout: DEFAULT_TIMEOUT,
            max_stdout_bytes: DEFAULT_MAX_STDOUT_BYTES,
            max_stderr_bytes: DEFAULT_MAX_STDERR_BYTES,
            sandbox: SandboxPolicy::Auto,
            network: NetworkPolicy::Off,
        }
    }
}

/// The `execute_code` tool.
pub struct ExecuteCodeTool {
    settings: ExecuteCodeSettings,
}

impl ExecuteCodeTool {
    /// A new tool with the given resolved settings.
    pub fn new(settings: ExecuteCodeSettings) -> Self {
        Self { settings }
    }
}

impl Default for ExecuteCodeTool {
    fn default() -> Self {
        Self::new(ExecuteCodeSettings::default())
    }
}

/// The tool's arguments: the Python `code`, plus an optional per-call `mode` override.
#[derive(Debug, Deserialize)]
struct ExecuteArgs {
    code: String,
    #[serde(default)]
    mode: Option<Mode>,
}

/// The structured detail attached to a result (opaque to the daemon; rendered by `kind`).
#[derive(Debug, Serialize)]
struct ExecDetail<'a> {
    status: &'a str,
    mode: &'a str,
    sandboxed: bool,
    /// The chosen backend label (`bwrap`/`landlock`/`sandbox-exec`/`plain`).
    backend: &'a str,
    exit_code: i32,
    duration_seconds: f64,
    stdout_len: usize,
    stderr_len: usize,
}

/// The tool-result JSON (hermes parity: `status` / `output` / `tool_calls_made` / `duration_seconds`
/// + optional `error`). `output` is omitted only for a setup failure (no process ran).
#[derive(Debug, Serialize)]
struct ResultJson {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
    tool_calls_made: u32,
    duration_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// The internal outcome of a completed run plus the chosen sandbox backend (for the detail envelope).
struct Executed {
    outcome: RunOutcome,
    kind: sandbox::SandboxKind,
    mode: Mode,
}

#[async_trait]
impl Tool for ExecuteCodeTool {
    fn name(&self) -> &str {
        "execute_code"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"code":{"type":"string","description":"Python source to execute in a sandboxed subprocess. Print your final result to stdout."},"mode":{"type":"string","enum":["project","strict"],"description":"project: run in the workspace with the resolved venv python. strict: isolated staging dir + system python. Defaults to the configured mode."}},"required":["code"]}"#
    }

    fn mutates(&self) -> bool {
        // Arbitrary code may write anywhere in the workspace; checkpoint before it runs.
        true
    }

    fn call_timeout(&self, _call: &ToolCall, _default: Option<Duration>) -> Option<Duration> {
        // execute_code manages its own deadline; opt out of the engine's per-tool timeout stage so
        // the two do not race (a double-kill or premature abort of a long-but-legitimate run).
        None
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let args: ExecuteArgs = match serde_json::from_str(&call.args) {
            Ok(args) => args,
            Err(e) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("execute_code: invalid arguments: {e}"),
                )
            }
        };
        if args.code.trim().is_empty() {
            return ToolOutcome::text(
                call.call_id.clone(),
                false,
                "execute_code: no code provided",
            );
        }
        let mode = args.mode.unwrap_or(self.settings.default_mode);

        // §12 approval gate — identical shape to the shell tool (policy-driven; durable defer parks).
        let prompt = approval_prompt(&args.code, mode);
        match approve_command(cx, prompt.clone()).await {
            Gate::Proceed => {}
            Gate::Reject(reason) => {
                return ToolOutcome::text(
                    call.call_id.clone(),
                    false,
                    format!("execute_code: {reason}"),
                )
            }
            Gate::Defer(job_id) => {
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

        let ws_root = cx.exec.cwd().to_path_buf();
        // Cluster E: on an untrusted (operator-bound) workspace root, project mode must not
        // auto-trust a workspace-discovered venv interpreter — thread the trust bit into resolution.
        let trusted = cx.exec.workspace_trusted();
        match self
            .execute(&ws_root, mode, trusted, &args.code, &cx.cancel)
            .await
        {
            Ok(exec) => self.success_outcome(call, exec),
            Err(e) => setup_error_outcome(call, &e.to_string()),
        }
    }
}

impl ExecuteCodeTool {
    /// Stage the script, resolve the interpreter + sandbox, run the subprocess, and clean up.
    async fn execute(
        &self,
        ws_root: &Path,
        mode: Mode,
        trusted: bool,
        code: &str,
        cancel: &CancellationToken,
    ) -> std::io::Result<Executed> {
        // Fail before staging when the interpreter or a required sandbox is unavailable, so a
        // setup error never leaves a stray staging dir behind.
        let interpreter = python::resolve_interpreter(mode, ws_root, trusted)
            .await
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "no usable python interpreter (>= 3.8) found",
                )
            })?;
        let kind = sandbox::resolve(self.settings.sandbox).await?;

        // Stage inside the workspace through `ContainedRoot` (openat2 RESOLVE_BENEATH |
        // RESOLVE_NO_SYMLINKS): the `.execute_code/<run_id>` dir and its `script.py` are created
        // fd-relative to the workspace root, so a symlinked `.execute_code` (or any symlinked
        // staging component) is refused rather than followed out of the workspace. `open` also
        // creates the root if missing (the historical lazy `ensure_root`).
        let rel_stage = Path::new(".execute_code").join(new_run_id());
        let rel_script = rel_stage.join("script.py");
        let root = ContainedRoot::open(ws_root)?;
        root.create_dir_all(&rel_stage).await?;
        root.write(&rel_script, code.as_bytes()).await?;

        // Absolute paths for the *child* (its argv + cwd) — resolved by the OS at spawn; the
        // daemon-side opens above are the containment-sensitive ones and never touch raw `fs`.
        let staging = ws_root.join(&rel_stage);
        let script = ws_root.join(&rel_script);
        let cwd = match mode {
            Mode::Project => ws_root.to_path_buf(),
            Mode::Strict => staging,
        };

        let run = self
            .run_staged(&script, &interpreter, &cwd, kind, cancel)
            .await;
        // Best-effort contained cleanup: `remove_dir_all_sync` unlinks a symlinked entry as the link
        // (never follows it out of root). Run off the reactor (blocking recursive remove).
        let ws = ws_root.to_path_buf();
        let _ = tokio::task::spawn_blocking(move || {
            ContainedRoot::open(&ws).and_then(|r| r.remove_dir_all_sync(&rel_stage))
        })
        .await;

        run.map(|outcome| Executed {
            outcome,
            kind,
            mode,
        })
    }

    /// Build the argv for the pre-staged `script` and run it to completion.
    async fn run_staged(
        &self,
        script: &Path,
        interpreter: &Path,
        cwd: &Path,
        kind: sandbox::SandboxKind,
        cancel: &CancellationToken,
    ) -> std::io::Result<RunOutcome> {
        let path_env = std::env::var_os("PATH").unwrap_or_default();
        let tz = std::env::var("TZ").ok().filter(|s| !s.is_empty());
        let argv = sandbox::argv(
            kind,
            self.settings.network,
            cwd,
            interpreter,
            script,
            &path_env,
            tz.as_deref(),
        );
        // The in-process (Landlock+seccomp) backend is applied at spawn; the argv-wrapper backends
        // (bwrap, sandbox-exec) carry their confinement in `argv` and pass no in-process spec.
        let confine = (kind == sandbox::SandboxKind::Landlock)
            .then(|| sandbox::landlock_spec(cwd, interpreter, self.settings.network));
        let caps = OutputCaps {
            stdout: self.settings.max_stdout_bytes,
            stderr: self.settings.max_stderr_bytes,
        };
        exec::run_subprocess(
            &argv,
            cwd,
            path_env,
            tz,
            self.settings.timeout,
            caps,
            confine,
            cancel,
        )
        .await
    }

    /// Assemble the tool result + detail from a completed run.
    fn success_outcome(&self, call: &ToolCall, exec: Executed) -> ToolOutcome {
        let RunOutcome {
            status,
            stdout,
            stderr,
            exit_code,
            duration,
        } = exec.outcome;
        let secs = round2(duration.as_secs_f64());
        let stdout_len = stdout.len();
        let stderr_len = stderr.len();

        let (status_label, output, error) = match status {
            Status::Success => ("success", stdout, None),
            Status::Timeout => {
                let msg = format!(
                    "Script timed out after {}s and was killed.",
                    self.settings.timeout.as_secs()
                );
                let out = if stdout.is_empty() {
                    format!("\u{23f0} {msg}")
                } else {
                    format!("{stdout}\n\n\u{23f0} {msg}")
                };
                ("timeout", out, Some(msg))
            }
            Status::Interrupted => (
                "interrupted",
                format!("{stdout}\n[execution interrupted]"),
                None,
            ),
            Status::Error => {
                let out = if stderr.is_empty() {
                    stdout
                } else {
                    format!("{stdout}\n--- stderr ---\n{stderr}")
                };
                let err = if stderr.is_empty() {
                    format!("Script exited with code {exit_code}")
                } else {
                    stderr.clone()
                };
                ("error", out, Some(err))
            }
        };

        let json = ResultJson {
            status: status_label,
            output: Some(output),
            tool_calls_made: 0,
            duration_seconds: secs,
            error,
        };
        let detail = ExecDetail {
            status: status_label,
            mode: exec.mode.as_str(),
            sandboxed: exec.kind.is_confined(),
            backend: exec.kind.label(),
            exit_code,
            duration_seconds: secs,
            stdout_len,
            stderr_len,
        };
        let ok = matches!(status, Status::Success);
        ToolOutcome::text(call.call_id.clone(), ok, encode(&json)).with_detail(ToolDetail {
            kind: "execute_code".into(),
            body: serde_json::to_vec(&detail).unwrap_or_default(),
        })
    }
}

/// A setup-failure result (no process ran): the hermes `{status:"error", error, ...}` shape.
fn setup_error_outcome(call: &ToolCall, msg: &str) -> ToolOutcome {
    let json = ResultJson {
        status: "error",
        output: None,
        tool_calls_made: 0,
        duration_seconds: 0.0,
        error: Some(format!("execute_code: {msg}")),
    };
    ToolOutcome::text(call.call_id.clone(), false, encode(&json))
}

/// Serialize a result to a compact JSON string (never fails for these plain structs).
fn encode(json: &ResultJson) -> String {
    serde_json::to_string(json).unwrap_or_else(|_| {
        r#"{"status":"error","tool_calls_made":0,"duration_seconds":0.0,"error":"execute_code: result serialization failed"}"#.to_string()
    })
}

/// The operator approval prompt: the mode + a bounded preview of the code.
fn approval_prompt(code: &str, mode: Mode) -> String {
    const PREVIEW: usize = 500;
    let preview: String = code.chars().take(PREVIEW).collect();
    let ellipsis = if code.chars().count() > PREVIEW {
        "\n… (truncated)"
    } else {
        ""
    };
    format!(
        "approve execute_code ({} mode, {} bytes):\n{preview}{ellipsis}",
        mode.as_str(),
        code.len()
    )
}

/// A process-unique run id (no external `uuid` dep): monotonic counter + wall-clock nanos.
fn new_run_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}-{n:x}")
}

/// Round to two decimals (hermes `round(x, 2)`).
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
