// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The execute_code subprocess runner (hermes local-path parity).
//!
//! Spawns the resolved argv with a scrubbed env (`env_clear` + `PATH` + `PYTHONDONTWRITEBYTECODE`,
//! plus `TZ` when set), drains stdout with a head+tail cap (40 % / 60 %) and stderr head-only, and
//! races the process against a self-managed timeout and the turn's cooperative cancel token. On
//! timeout or cancel it kills the whole process group (SIGTERM, then SIGKILL after 5 s) so a script
//! that spawned children cannot outlive the tool call.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::path::Path;
use std::process::{ExitStatus, Stdio};
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio_util::sync::CancellationToken;

/// The grace period between SIGTERM and SIGKILL when terminating a runaway child (hermes: 5 s).
const KILL_GRACE: Duration = Duration::from_secs(5);
/// How long to wait for the output readers to finish draining after the process ends.
const READER_JOIN: Duration = Duration::from_secs(3);
/// The read buffer size for draining a child pipe.
const READ_CHUNK: usize = 4096;

/// A drained pipe: `(head, tail, total_bytes_seen)`.
type Drained = (Vec<u8>, Vec<u8>, usize);

/// The byte caps applied to captured output.
pub(crate) struct OutputCaps {
    /// Total stdout budget (split 40 % head / 60 % tail).
    pub stdout: usize,
    /// Total stderr budget (head-only).
    pub stderr: usize,
}

/// The classified outcome of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    /// The process exited 0.
    Success,
    /// The process exited non-zero (or its status could not be read).
    Error,
    /// The self-managed timeout fired and the process was killed.
    Timeout,
    /// The turn's cancel token fired and the process was killed.
    Interrupted,
}

/// A completed run: the classified status, capped+decoded output, exit code, and wall-clock time.
pub(crate) struct RunOutcome {
    /// The classified status.
    pub status: Status,
    /// Captured stdout (head+tail assembled, decoded lossily).
    pub stdout: String,
    /// Captured stderr (head-only, decoded lossily).
    pub stderr: String,
    /// The process exit code (`-1` when killed by signal or unavailable).
    pub exit_code: i32,
    /// Wall-clock duration from spawn to completion.
    pub duration: Duration,
}

/// Spawn `argv` in `cwd` and run it to completion, honoring `timeout` and `cancel`.
///
/// When `confine` is `Some`, the in-process (Linux Landlock+seccomp) sandbox is installed on the
/// child at spawn (a no-op off Linux); `TMPDIR` is redirected under the working dir so the child's
/// temp files stay inside the sandbox's writable scope. Argv-wrapper backends (bwrap, `sandbox-exec`)
/// carry their confinement in `argv` and pass `confine = None`.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_subprocess(
    argv: &[OsString],
    cwd: &Path,
    path_env: OsString,
    tz: Option<String>,
    timeout: Duration,
    caps: OutputCaps,
    confine: Option<daemon_sandbox::SandboxSpec>,
    cancel: &CancellationToken,
) -> std::io::Result<RunOutcome> {
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .current_dir(cwd)
        .env_clear()
        .env("PATH", &path_env)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(tz) = &tz {
        cmd.env("TZ", tz);
    }
    if let Some(spec) = &confine {
        // Keep the child's temp files inside the sandbox's writable scope (Landlock cannot grant a
        // private /tmp), then install the in-process confinement before spawn.
        cmd.env("TMPDIR", cwd);
        daemon_sandbox::confine_command(&mut cmd, spec)?;
    }
    // A fresh process group so the timeout/cancel path can signal the whole tree (the child plus any
    // grandchildren it spawns), not just the direct child.
    #[cfg(unix)]
    cmd.process_group(0);

    let start = Instant::now();
    let mut child = cmd.spawn()?;
    let pid = child.id();

    // Drain both pipes concurrently on their own tasks so a large writer cannot deadlock on a full
    // pipe buffer while we wait.
    let head_cap = caps.stdout * 2 / 5; // 40 % head
    let tail_cap = caps.stdout - head_cap; // 60 % tail
    let stdout_reader = child
        .stdout
        .take()
        .map(|p| tokio::spawn(drain(p, head_cap, tail_cap)));
    let stderr_reader = child
        .stderr
        .take()
        .map(|p| tokio::spawn(drain(p, caps.stderr, 0)));

    let (status, exit_status) = tokio::select! {
        res = child.wait() => match res {
            Ok(st) if st.success() => (Status::Success, Some(st)),
            Ok(st) => (Status::Error, Some(st)),
            Err(_) => (Status::Error, None),
        },
        _ = tokio::time::sleep(timeout) => (Status::Timeout, terminate(&mut child, pid).await),
        _ = cancel.cancelled() => (Status::Interrupted, terminate(&mut child, pid).await),
    };

    let (out_head, out_tail, out_total) = join_reader(stdout_reader).await;
    let (err_head, _err_tail, _err_total) = join_reader(stderr_reader).await;

    let stdout = assemble_stdout(&out_head, &out_tail, out_total, caps.stdout);
    let stderr = String::from_utf8_lossy(&err_head).into_owned();
    let exit_code = exit_status.and_then(|s| s.code()).unwrap_or(-1);

    Ok(RunOutcome {
        status,
        stdout,
        stderr,
        exit_code,
        duration: start.elapsed(),
    })
}

/// Kill the child's process group (SIGTERM, then SIGKILL after [`KILL_GRACE`]) and reap it.
#[cfg(unix)]
async fn terminate(child: &mut Child, pid: Option<u32>) -> Option<ExitStatus> {
    use nix::sys::signal::{killpg, Signal};
    use nix::unistd::Pid;

    let pgid = pid.and_then(|p| i32::try_from(p).ok()).map(Pid::from_raw);
    match pgid {
        Some(pgid) => {
            let _ = killpg(pgid, Signal::SIGTERM);
        }
        None => {
            let _ = child.start_kill();
        }
    }
    match tokio::time::timeout(KILL_GRACE, child.wait()).await {
        Ok(Ok(st)) => return Some(st),
        Ok(Err(_)) => return None,
        Err(_) => {}
    }
    if let Some(pgid) = pgid {
        let _ = killpg(pgid, Signal::SIGKILL);
    }
    child.wait().await.ok()
}

/// Non-Unix fallback: kill just the direct child (no process-group semantics).
#[cfg(not(unix))]
async fn terminate(child: &mut Child, _pid: Option<u32>) -> Option<ExitStatus> {
    let _ = child.start_kill();
    match tokio::time::timeout(KILL_GRACE, child.wait()).await {
        Ok(Ok(st)) => Some(st),
        _ => {
            let _ = child.kill().await;
            child.wait().await.ok()
        }
    }
}

/// Await a reader task with a bounded join so a stuck pipe never wedges the tool.
async fn join_reader(handle: Option<tokio::task::JoinHandle<Drained>>) -> Drained {
    let Some(handle) = handle else {
        return (Vec::new(), Vec::new(), 0);
    };
    match tokio::time::timeout(READER_JOIN, handle).await {
        Ok(Ok(collected)) => collected,
        _ => (Vec::new(), Vec::new(), 0),
    }
}

/// Read `pipe` to EOF, keeping the first `head_cap` bytes and a rolling last `tail_cap` bytes.
/// Returns `(head, tail, total)` where `total` is the full byte count seen (for the truncation note).
async fn drain<R: AsyncRead + Unpin>(mut pipe: R, head_cap: usize, tail_cap: usize) -> Drained {
    let mut acc = HeadTail::new(head_cap, tail_cap);
    let mut buf = [0u8; READ_CHUNK];
    loop {
        match pipe.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => acc.push(&buf[..n]),
        }
    }
    acc.finish()
}

/// A bounded head + rolling-tail byte accumulator: keeps the first `head_cap` bytes and the last
/// `tail_cap` bytes seen (byte-granular, so it is correct even when one write exceeds `tail_cap`).
struct HeadTail {
    head_cap: usize,
    tail_cap: usize,
    head: Vec<u8>,
    tail: VecDeque<u8>,
    total: usize,
}

impl HeadTail {
    fn new(head_cap: usize, tail_cap: usize) -> Self {
        Self {
            head_cap,
            tail_cap,
            head: Vec::new(),
            tail: VecDeque::new(),
            total: 0,
        }
    }

    fn push(&mut self, data: &[u8]) {
        self.total += data.len();
        let mut rest = data;
        if self.head.len() < self.head_cap {
            let take = (self.head_cap - self.head.len()).min(rest.len());
            self.head.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
        }
        if rest.is_empty() || self.tail_cap == 0 {
            return;
        }
        self.tail.extend(rest.iter().copied());
        // Keep only the most-recent `tail_cap` bytes. Each read is bounded by READ_CHUNK, so the
        // deque overshoots by at most one chunk before trimming (bounded memory).
        while self.tail.len() > self.tail_cap {
            self.tail.pop_front();
        }
    }

    fn finish(self) -> Drained {
        (self.head, self.tail.into_iter().collect(), self.total)
    }
}

/// Assemble captured stdout, inserting a truncation notice when the total exceeded the cap.
fn assemble_stdout(head: &[u8], tail: &[u8], total: usize, cap: usize) -> String {
    if total > cap && !tail.is_empty() {
        let omitted = total.saturating_sub(head.len()).saturating_sub(tail.len());
        format!(
            "{}\n\n... [OUTPUT TRUNCATED - {} chars omitted out of {} total] ...\n\n{}",
            String::from_utf8_lossy(head),
            commafy(omitted),
            commafy(total),
            String::from_utf8_lossy(tail),
        )
    } else {
        let mut s = String::from_utf8_lossy(head).into_owned();
        s.push_str(&String::from_utf8_lossy(tail));
        s
    }
}

/// Format an integer with thousands separators (hermes uses `{n:,}` in the truncation note).
fn commafy(n: usize) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commafy_groups_thousands() {
        assert_eq!(commafy(0), "0");
        assert_eq!(commafy(42), "42");
        assert_eq!(commafy(1_234), "1,234");
        assert_eq!(commafy(1_234_567), "1,234,567");
    }

    #[test]
    fn head_tail_keeps_ends_and_counts_total() {
        let mut ht = HeadTail::new(4, 4);
        for _ in 0..10 {
            ht.push(b"AB"); // 20 bytes total
        }
        let (head, tail, total) = ht.finish();
        assert_eq!(total, 20);
        assert_eq!(head, b"ABAB");
        // Tail keeps the most-recent bytes within budget (chunk-granular eviction).
        assert!(tail.ends_with(b"AB"));
        assert!(tail.len() <= 4);
    }

    #[test]
    fn head_only_when_tail_cap_zero() {
        let mut ht = HeadTail::new(3, 0);
        ht.push(b"abcdef");
        let (head, tail, total) = ht.finish();
        assert_eq!(head, b"abc");
        assert!(tail.is_empty());
        assert_eq!(total, 6);
    }

    #[test]
    fn assemble_inserts_notice_only_when_over_cap() {
        let full = assemble_stdout(b"abc", b"", 3, 10);
        assert_eq!(full, "abc");
        let truncated = assemble_stdout(b"aa", b"zz", 100, 4);
        assert!(truncated.contains("OUTPUT TRUNCATED"));
        assert!(truncated.contains("96 chars omitted"));
        assert!(truncated.starts_with("aa"));
        assert!(truncated.ends_with("zz"));
    }
}
