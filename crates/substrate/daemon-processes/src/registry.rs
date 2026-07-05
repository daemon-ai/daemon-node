// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The process registry: spawn, track, and manage host-owned background OS processes (hermes
//! `ProcessRegistry`). One resident instance per node, shared by the `shell` tool (spawn), the
//! `process` tool (query/control), and the exit notifier.
//!
//! Lock order (where both are held): a session's `state` lock before the registry `global`
//! breaker lock. The `tracked` map lock never nests inside a session lock.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use daemon_common::SessionId;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::ansi::strip_ansi;
use crate::clock::Clock;
use crate::ring::RingBuffer;
use crate::watch::{GlobalBreaker, ScanOutcome, WatchState};
use crate::{ProcessNotifier, RegistryConfig};

/// How long a direct child may be observed exited with the reader thread still running before
/// [`ProcessRegistry::reconcile`] force-flips the session (the orphaned-pipe guard, hermes issue
/// #17327). The grace keeps the healthy path intact: a normal reader reaches EOF, appends the
/// final chunk, and marks the exit itself within this window.
const RECONCILE_GRACE_MS: u64 = 500;

/// Shell startup noise stripped from the head of a process's output (hermes
/// `_SHELL_NOISE_SUBSTRINGS`).
const SHELL_NOISE: &[&str] = &[
    "bash: cannot set terminal process group",
    "bash: no job control in this shell",
    "no job control in this shell",
    "cannot set terminal process group",
    "tcsetattr: Inappropriate ioctl for device",
];

/// Why a tracked process stopped being live. (Hermes also has `failed_start`/`lost` for its
/// sandbox backends; the daemon's only backend is local, and a failed spawn is a [`SpawnError`]
/// returned to the tool rather than a tracked session.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionReason {
    /// The process ran to completion on its own.
    Exited,
    /// It was terminated through the registry (`process(kill)`, shutdown).
    Killed,
}

impl CompletionReason {
    /// The wire string (hermes `completion_reason`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exited => "exited",
            Self::Killed => "killed",
        }
    }
}

/// The live backend handle of a tracked process.
enum Backend {
    /// A `sh -c` child with merged stdout+stderr on a pipe (stdin is null — hermes parity: pipe
    /// sessions take no stdin; use `pty` for interactive tools).
    Piped(Arc<shared_child::SharedChild>),
    /// A PTY session: interactive CLIs read/write the pty; stdin actions work here.
    Pty(PtyBackend),
    /// No runtime handle (a synthetic/recovered session): status-only, kill unavailable.
    #[allow(
        dead_code,
        reason = "constructed by tests; mirrors hermes' detached sessions"
    )]
    Detached,
}

struct PtyBackend {
    child: Mutex<Box<dyn portable_pty::Child + Send + Sync>>,
    /// The pty writer for stdin write/submit; `close` sends EOT (^D) — a pty has no pipe EOF.
    writer: Mutex<Option<Box<dyn Write + Send>>>,
    /// Held so the pty pair outlives the child; released at exit.
    master: Mutex<Option<Box<dyn portable_pty::MasterPty + Send>>>,
}

/// Mutable per-process state, updated by the reader thread and the control paths.
struct ProcState {
    exited: bool,
    exit_code: Option<i32>,
    reason: CompletionReason,
    termination_source: Option<String>,
    output: RingBuffer,
    saw_output: bool,
    watch: WatchState,
    watch_patterns: Vec<String>,
    notify_on_complete: bool,
}

/// One tracked background process.
pub struct ProcSession {
    id: String,
    owner: SessionId,
    command: String,
    cwd: PathBuf,
    pid: Option<u32>,
    started_unix: u64,
    backend: Backend,
    state: Mutex<ProcState>,
    done: tokio::sync::Notify,
    /// Live `wait` callers: an exit observed by a waiter suppresses the duplicate completion
    /// notification (hermes' consumed-completion drain skip).
    waiters: AtomicUsize,
    /// When [`ProcessRegistry::reconcile`] first observed the direct child exited while the reader
    /// had not yet completed (monotonic ms). The flip is forced only after a grace window, so the
    /// reader's normal EOF → wait → mark path is never raced out of the final output chunk.
    reconcile_armed_ms: Mutex<Option<u64>>,
}

impl ProcSession {
    /// The registry-assigned `proc_<12hex>` id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The session that spawned this process (ownership scope).
    pub fn owner(&self) -> &SessionId {
        &self.owner
    }

    /// The OS pid, when the spawn produced one.
    pub fn pid(&self) -> Option<u32> {
        self.pid
    }
}

/// A spawn request from the shell tool (already validated/contained by the caller).
pub struct SpawnRequest {
    /// The owning session (notifications route back here; `process` actions are scoped to it).
    pub owner: SessionId,
    /// The shell command line to run under `sh -c` (background/PTY are the approved shell-string
    /// surface; the denylist/approval gates already ran on it).
    pub line: String,
    /// The working directory (absolute, already contained by the shell tool).
    pub cwd: PathBuf,
    /// Spawn on a pseudo-terminal (interactive CLIs; enables the stdin actions).
    pub pty: bool,
    /// Queue exactly one notification into the owner when the process exits.
    pub notify_on_complete: bool,
    /// Watch patterns (rate-limited; mutually exclusive with `notify_on_complete` at the tool).
    pub watch_patterns: Vec<String>,
}

/// A spawn failure (the tool reports it; nothing is tracked).
#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    /// The OS refused the spawn.
    #[error("failed to start background process: {0}")]
    Io(#[from] std::io::Error),
    /// The PTY layer refused the spawn.
    #[error("failed to start pty process: {0}")]
    Pty(String),
}

/// `process(poll)` — status + fresh output preview (hermes `poll`).
#[derive(Debug, Serialize)]
pub struct PollResult {
    /// The `proc_` id.
    pub session_id: String,
    /// The original command line.
    pub command: String,
    /// `"running"` / `"exited"`.
    pub status: &'static str,
    /// The OS pid, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    /// Seconds since spawn.
    pub uptime_seconds: u64,
    /// The last 1000 bytes of output, ANSI-stripped.
    pub output_preview: String,
    /// Exit code once exited.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Completion reason once exited.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_reason: Option<&'static str>,
    /// What terminated it (kill source), when killed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub termination_source: Option<String>,
}

/// `process(log)` — paged full output (hermes `read_log`).
#[derive(Debug, Serialize)]
pub struct LogResult {
    /// The `proc_` id.
    pub session_id: String,
    /// `"running"` / `"exited"`.
    pub status: &'static str,
    /// The selected lines, ANSI-stripped.
    pub output: String,
    /// Total retained lines.
    pub total_lines: usize,
    /// How many lines this page shows.
    pub showing: String,
}

/// How a `process(wait)` resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitStatus {
    /// The process exited within the wait window.
    Exited,
    /// The wait window elapsed with the process still running.
    Timeout,
    /// The turn was cancelled while waiting.
    Interrupted,
}

/// `process(wait)` — block-until-exit outcome (hermes `wait`).
#[derive(Debug, Serialize)]
pub struct WaitResult {
    /// How the wait resolved.
    pub status: WaitStatus,
    /// Exit code when `Exited`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Completion reason when `Exited`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_reason: Option<&'static str>,
    /// Kill source when `Exited` by kill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub termination_source: Option<String>,
    /// Output tail (2000 bytes on exit, 1000 on timeout/interrupt), ANSI-stripped.
    pub output: String,
}

/// `process(kill)` outcome.
#[derive(Debug, PartialEq, Eq)]
pub enum KillResult {
    /// No such process.
    NotFound,
    /// It had already finished.
    AlreadyExited {
        /// Its recorded exit code.
        exit_code: Option<i32>,
    },
    /// Terminated (SIGTERM to the process group).
    Killed,
    /// The kill could not be delivered.
    Failed(String),
}

/// `process(write|submit|close)` outcome.
#[derive(Debug, PartialEq, Eq)]
pub enum StdinResult {
    /// No such process.
    NotFound,
    /// It had already finished.
    AlreadyExited,
    /// This session has no writable stdin (piped sessions run with stdin null — hermes parity;
    /// spawn with `pty=true` for interactive input).
    NotAvailable,
    /// Written/flushed.
    Ok {
        /// Bytes written (0 for `close`).
        bytes: usize,
    },
    /// The write failed.
    Failed(String),
}

/// One row of `process(list)` (hermes `list_sessions`).
#[derive(Debug, Serialize)]
pub struct ProcSummary {
    /// The `proc_` id.
    pub session_id: String,
    /// The command line (truncated to 200 chars).
    pub command: String,
    /// The working directory.
    pub cwd: String,
    /// The OS pid, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    /// Unix seconds at spawn.
    pub started_unix: u64,
    /// Seconds since spawn.
    pub uptime_seconds: u64,
    /// `"running"` / `"exited"`.
    pub status: &'static str,
    /// The last 200 bytes of output, ANSI-stripped.
    pub output_preview: String,
    /// Exit code once exited.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
}

struct Tracked {
    running: HashMap<String, Arc<ProcSession>>,
    finished: HashMap<String, Arc<ProcSession>>,
}

/// The resident background-process registry. Construct once at node assembly (never per turn:
/// background processes outlive the turn that spawned them, and a turn cancel must not kill them).
pub struct ProcessRegistry {
    cfg: RegistryConfig,
    clock: Arc<dyn Clock>,
    tracked: Mutex<Tracked>,
    global: Mutex<GlobalBreaker>,
    /// Processes whose completion the agent already observed via wait/poll/log — their completion
    /// notification is suppressed (hermes `_completion_consumed`).
    consumed: Mutex<HashSet<String>>,
    /// The per-session sticky working directory (`cd` persistence for the shell tool). Ephemeral
    /// by design (decision: in-memory v1).
    session_cwd: Mutex<HashMap<SessionId, PathBuf>>,
    notifier: OnceLock<Arc<dyn ProcessNotifier>>,
    /// The runtime notifications are delivered on, captured at [`set_notifier`](Self::set_notifier).
    runtime: Mutex<Option<tokio::runtime::Handle>>,
    salt: AtomicU64,
}

impl ProcessRegistry {
    /// A registry over `cfg` and the injected `clock`.
    pub fn new(cfg: RegistryConfig, clock: Arc<dyn Clock>) -> Self {
        Self {
            cfg,
            clock,
            tracked: Mutex::new(Tracked {
                running: HashMap::new(),
                finished: HashMap::new(),
            }),
            global: Mutex::new(GlobalBreaker::default()),
            consumed: Mutex::new(HashSet::new()),
            session_cwd: Mutex::new(HashMap::new()),
            notifier: OnceLock::new(),
            runtime: Mutex::new(None),
            salt: AtomicU64::new(0),
        }
    }

    /// The registry's shell/process limits (read by the tools for clamps + truncation).
    pub fn config(&self) -> &RegistryConfig {
        &self.cfg
    }

    /// Late-bind the notification sink (the node's session-inject adapter) and capture the current
    /// tokio runtime for delivery. Idempotent: the first set wins. Must be called on the runtime
    /// notifications should be delivered on (node assembly is).
    pub fn set_notifier(&self, notifier: Arc<dyn ProcessNotifier>) {
        if self.notifier.set(notifier).is_ok() {
            *self.runtime.lock().unwrap() = tokio::runtime::Handle::try_current().ok();
        }
    }

    /// Deliver one formatted notification to `owner`, on the captured runtime. Dropped (debug-
    /// logged) when no notifier/runtime is bound — a node assembled without the seam.
    fn emit(&self, owner: &SessionId, text: String) {
        let (Some(notifier), Some(handle)) = (
            self.notifier.get().cloned(),
            self.runtime.lock().unwrap().clone(),
        ) else {
            tracing::debug!(owner = %owner, "dropping process notification (no notifier bound)");
            return;
        };
        let owner = owner.clone();
        handle.spawn(async move {
            notifier.notify(&owner, text).await;
        });
    }

    /// Mint a fresh `proc_<12hex>` id (nanos ⊕ a process-unique salt, hermes id shape).
    fn gen_id(&self) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let salt = self.salt.fetch_add(1, Ordering::Relaxed);
        format!(
            "proc_{:012x}",
            (nanos ^ salt.rotate_left(48)) & 0xFFFF_FFFF_FFFF
        )
    }

    // ----- session cwd (shell `cd` persistence) -----

    /// The sticky working directory recorded for `session`, if any.
    pub fn cwd_for(&self, session: &SessionId) -> Option<PathBuf> {
        self.session_cwd.lock().unwrap().get(session).cloned()
    }

    /// Record `session`'s sticky working directory (already contained by the shell tool).
    pub fn set_cwd(&self, session: &SessionId, dir: PathBuf) {
        self.session_cwd
            .lock()
            .unwrap()
            .insert(session.clone(), dir);
    }

    // ----- spawn -----

    /// Spawn a background process (piped or PTY). The command line runs under `sh -c` with
    /// `set +m` and merged stderr, in its own process group (so kill reaches the whole tree), with
    /// a scrubbed child environment (PATH + PYTHONUNBUFFERED — the [`LocalEnvironment`] invariant:
    /// no inherited host secrets).
    ///
    /// [`LocalEnvironment`]: https://docs.rs/daemon-core
    pub fn spawn(self: &Arc<Self>, req: SpawnRequest) -> Result<Arc<ProcSession>, SpawnError> {
        std::fs::create_dir_all(&req.cwd)?;
        let id = self.gen_id();
        let state = ProcState {
            exited: false,
            exit_code: None,
            reason: CompletionReason::Exited,
            termination_source: None,
            output: RingBuffer::new(self.cfg.max_output_bytes),
            saw_output: false,
            watch: WatchState::default(),
            watch_patterns: req.watch_patterns,
            notify_on_complete: req.notify_on_complete,
        };

        let session = if req.pty {
            self.spawn_pty(&id, &req.line, &req.cwd, req.owner, state)?
        } else {
            self.spawn_piped(&id, &req.line, &req.cwd, req.owner, state)?
        };

        {
            let mut tracked = self.tracked.lock().unwrap();
            let mut consumed = self.consumed.lock().unwrap();
            self.prune_locked(&mut tracked, &mut consumed);
            tracked.running.insert(id, session.clone());
        }
        Ok(session)
    }

    fn spawn_piped(
        self: &Arc<Self>,
        id: &str,
        line: &str,
        cwd: &std::path::Path,
        owner: SessionId,
        state: ProcState,
    ) -> Result<Arc<ProcSession>, SpawnError> {
        // `exec 2>&1` merges stderr into the one captured stream with true ordering; `set +m`
        // silences job-control noise (hermes parity).
        let script = format!("exec 2>&1\nset +m\n{line}");
        let mut command = std::process::Command::new("sh");
        command
            .arg("-c")
            .arg(script)
            .current_dir(cwd)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("PYTHONUNBUFFERED", "1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // Own process group: kill = killpg(pid) reaches every descendant.
            command.process_group(0);
        }
        let child = shared_child::SharedChild::spawn(&mut command)?;
        let stdout = child.take_stdout();
        let child = Arc::new(child);

        let session = Arc::new(ProcSession {
            id: id.to_string(),
            owner,
            command: line.to_string(),
            cwd: cwd.to_path_buf(),
            pid: Some(child.id()),
            started_unix: self.clock.now_unix(),
            backend: Backend::Piped(child.clone()),
            state: Mutex::new(state),
            done: tokio::sync::Notify::new(),
            waiters: AtomicUsize::new(0),
            reconcile_armed_ms: Mutex::new(None),
        });

        let registry = self.clone();
        let reader_session = session.clone();
        let chunk = self.cfg.reader_chunk_bytes.max(1);
        std::thread::Builder::new()
            .name(format!("proc-reader-{id}"))
            .spawn(move || {
                if let Some(mut stdout) = stdout {
                    let mut buf = vec![0u8; chunk];
                    loop {
                        match stdout.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => registry.ingest_chunk(&reader_session, &buf[..n]),
                        }
                    }
                }
                // Always reap the child (no zombies), then flip to finished exactly once.
                let code = child.wait().ok().and_then(|status| status.code());
                registry.mark_exited(&reader_session, code, CompletionReason::Exited, None);
            })?;
        Ok(session)
    }

    fn spawn_pty(
        self: &Arc<Self>,
        id: &str,
        line: &str,
        cwd: &std::path::Path,
        owner: SessionId,
        state: ProcState,
    ) -> Result<Arc<ProcSession>, SpawnError> {
        use portable_pty::{native_pty_system, CommandBuilder, PtySize};

        let pty = native_pty_system();
        let pair = pty
            .openpty(PtySize {
                rows: self.cfg.pty_rows,
                cols: self.cfg.pty_cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| SpawnError::Pty(e.to_string()))?;

        let mut builder = CommandBuilder::new("sh");
        builder.arg("-c");
        builder.arg(format!("set +m; {line}"));
        builder.cwd(cwd);
        builder.env_clear();
        builder.env("PATH", std::env::var_os("PATH").unwrap_or_default());
        builder.env("PYTHONUNBUFFERED", "1");
        builder.env("TERM", "xterm-256color");

        let child = pair
            .slave
            .spawn_command(builder)
            .map_err(|e| SpawnError::Pty(e.to_string()))?;
        // Drop the slave: the child holds its own copy; keeping ours would hold the pty open.
        drop(pair.slave);
        let pid = child.process_id();
        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| SpawnError::Pty(e.to_string()))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|e| SpawnError::Pty(e.to_string()))?;

        let session = Arc::new(ProcSession {
            id: id.to_string(),
            owner,
            command: line.to_string(),
            cwd: cwd.to_path_buf(),
            pid,
            started_unix: self.clock.now_unix(),
            backend: Backend::Pty(PtyBackend {
                child: Mutex::new(child),
                writer: Mutex::new(Some(writer)),
                master: Mutex::new(Some(pair.master)),
            }),
            state: Mutex::new(state),
            done: tokio::sync::Notify::new(),
            waiters: AtomicUsize::new(0),
            reconcile_armed_ms: Mutex::new(None),
        });

        let registry = self.clone();
        let reader_session = session.clone();
        let chunk = self.cfg.reader_chunk_bytes.max(1);
        std::thread::Builder::new()
            .name(format!("proc-pty-reader-{id}"))
            .spawn(move || {
                let mut reader = reader;
                let mut buf = vec![0u8; chunk];
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => registry.ingest_chunk(&reader_session, &buf[..n]),
                    }
                }
                let code = match &reader_session.backend {
                    Backend::Pty(pty) => pty
                        .child
                        .lock()
                        .unwrap()
                        .wait()
                        .ok()
                        .map(|status| i32::try_from(status.exit_code()).unwrap_or(-1)),
                    _ => None,
                };
                registry.mark_exited(&reader_session, code, CompletionReason::Exited, None);
            })?;
        Ok(session)
    }

    // ----- reader-side ingestion (output ring + watch patterns) -----

    fn ingest_chunk(&self, session: &Arc<ProcSession>, chunk: &[u8]) {
        let text = String::from_utf8_lossy(chunk).into_owned();
        let scan;
        {
            let mut st = session.state.lock().unwrap();
            // Suppress-after-exit: late pipe chunks after the process was declared exited are
            // post-exit noise (hermes parity) — never re-scanned for watch patterns.
            if st.exited {
                st.output.append(chunk);
                return;
            }
            if !st.saw_output {
                st.saw_output = true;
                let cleaned = clean_shell_noise(&text);
                st.output.append(cleaned.as_bytes());
            } else {
                st.output.append(chunk);
            }
            let patterns = st.watch_patterns.clone();
            scan = st.watch.scan(
                self.clock.now_ms(),
                self.cfg.watch_min_interval_secs * 1000,
                self.cfg.watch_strike_limit,
                &patterns,
                &text,
            );
            if matches!(scan, Some(ScanOutcome::Disabled { .. })) {
                // Promote: the agent still gets exactly one notification when the process ends.
                st.notify_on_complete = true;
            }
        }
        match scan {
            Some(ScanOutcome::Emit {
                pattern,
                output,
                suppressed,
            }) => {
                // The cross-session breaker sits after the per-session limit (hermes ordering).
                let outcome = self.global.lock().unwrap().admit(
                    self.clock.now_ms(),
                    self.cfg.watch_global_max_per_window,
                    self.cfg.watch_global_window_secs * 1000,
                    self.cfg.watch_global_cooldown_secs * 1000,
                );
                if let Some(suppressed) = outcome.released {
                    self.emit(
                        &session.owner,
                        format!(
                            "[IMPORTANT: Watch-pattern notifications resumed. {suppressed} match \
                             event(s) were suppressed during the flood.]"
                        ),
                    );
                }
                if outcome.tripped {
                    self.emit(
                        &session.owner,
                        format!(
                            "[IMPORTANT: Watch-pattern overflow: >{} notifications in {}s across \
                             all processes. Suppressing further watch_match events for {}s.]",
                            self.cfg.watch_global_max_per_window,
                            self.cfg.watch_global_window_secs,
                            self.cfg.watch_global_cooldown_secs
                        ),
                    );
                }
                if outcome.admitted {
                    self.emit(
                        &session.owner,
                        format_watch_match(
                            &session.id,
                            &session.command,
                            &pattern,
                            &output,
                            suppressed,
                        ),
                    );
                }
            }
            Some(ScanOutcome::Disabled { .. }) => {
                self.emit(
                    &session.owner,
                    format!(
                        "[IMPORTANT: Watch patterns disabled for process {} — {} consecutive \
                         rate-limit windows triggered (min spacing {}s). Falling back to \
                         notify_on_complete semantics; you'll get exactly one notification when \
                         the process exits.]",
                        session.id, self.cfg.watch_strike_limit, self.cfg.watch_min_interval_secs
                    ),
                );
            }
            Some(ScanOutcome::Dropped) | None => {}
        }
    }

    /// Flip a session to exited exactly once (idempotent — the kill path and the reader thread may
    /// both arrive here, hermes `_move_to_finished`), move it running → finished, wake waiters, and
    /// queue the completion notification when armed and not already consumed by a live waiter.
    fn mark_exited(
        &self,
        session: &Arc<ProcSession>,
        exit_code: Option<i32>,
        reason: CompletionReason,
        termination_source: Option<String>,
    ) {
        let (notify, tail);
        {
            let mut st = session.state.lock().unwrap();
            if st.exited {
                return;
            }
            st.exited = true;
            st.exit_code = exit_code;
            st.reason = reason;
            st.termination_source = termination_source;
            notify = st.notify_on_complete;
            tail = strip_ansi(&st.output.tail_lossy(2000));
        }
        // Release pty handles so the fd pair closes promptly.
        if let Backend::Pty(pty) = &session.backend {
            *pty.writer.lock().unwrap() = None;
            *pty.master.lock().unwrap() = None;
        }
        let was_running = {
            let mut tracked = self.tracked.lock().unwrap();
            match tracked.running.remove(&session.id) {
                Some(live) => {
                    tracked.finished.insert(session.id.clone(), live);
                    true
                }
                None => false,
            }
        };
        session.done.notify_waiters();

        // One notification per completion: only on the first running→finished move, only when no
        // live `wait` observes the exit directly, and never after a poll/log/wait already consumed
        // it (hermes' drain-skip semantics).
        let consumed = self.consumed.lock().unwrap().contains(&session.id);
        if was_running && notify && session.waiters.load(Ordering::SeqCst) == 0 && !consumed {
            let st = session.state.lock().unwrap();
            self.emit(
                &session.owner,
                format_completion(
                    &session.id,
                    &session.command,
                    st.exit_code,
                    st.reason,
                    st.termination_source.as_deref(),
                    &tail,
                ),
            );
        }
    }

    /// Reconcile a session against its real child state: the direct child may have exited while a
    /// descendant still holds the output pipe open, leaving the reader blocked forever (hermes
    /// issue #17327 — poll reported "running" indefinitely). Poll/wait call this; after a short
    /// grace window (so the healthy reader-EOF path always wins the race and no final output
    /// chunk is lost) the session is force-flipped to exited.
    fn reconcile(&self, session: &Arc<ProcSession>) {
        if session.state.lock().unwrap().exited {
            return;
        }
        let code = match &session.backend {
            Backend::Piped(child) => match child.try_wait() {
                Ok(Some(status)) => Some(status.code()),
                _ => None,
            },
            Backend::Pty(pty) => match pty.child.lock().unwrap().try_wait() {
                Ok(Some(status)) => Some(Some(i32::try_from(status.exit_code()).unwrap_or(-1))),
                _ => None,
            },
            Backend::Detached => None,
        };
        let Some(code) = code else {
            return;
        };
        let now = self.clock.now_ms();
        let armed = {
            let mut armed = session.reconcile_armed_ms.lock().unwrap();
            *armed.get_or_insert(now)
        };
        if now.saturating_sub(armed) < RECONCILE_GRACE_MS {
            return; // grace: give the reader its EOF → final-chunk → mark path
        }
        tracing::info!(
            proc = %session.id,
            "direct child exited but the reader is still blocked (orphaned pipe); reconciling"
        );
        self.mark_exited(session, code, CompletionReason::Exited, None);
    }

    fn get(&self, id: &str) -> Option<Arc<ProcSession>> {
        let tracked = self.tracked.lock().unwrap();
        tracked
            .running
            .get(id)
            .or_else(|| tracked.finished.get(id))
            .cloned()
    }

    /// The owning session of `id`, if tracked (the tools' ownership gate).
    pub fn owner_of(&self, id: &str) -> Option<SessionId> {
        self.get(id).map(|s| s.owner.clone())
    }

    // ----- queries -----

    /// Status + fresh output preview; marks an observed completion consumed.
    pub fn poll(&self, id: &str) -> Option<PollResult> {
        let session = self.get(id)?;
        self.reconcile(&session);
        let st = session.state.lock().unwrap();
        if st.exited {
            self.consumed.lock().unwrap().insert(session.id.clone());
        }
        Some(PollResult {
            session_id: session.id.clone(),
            command: session.command.clone(),
            status: if st.exited { "exited" } else { "running" },
            pid: session.pid,
            uptime_seconds: self.clock.now_unix().saturating_sub(session.started_unix),
            output_preview: strip_ansi(&st.output.tail_lossy(1000)),
            exit_code: if st.exited { st.exit_code } else { None },
            completion_reason: st.exited.then(|| st.reason.as_str()),
            termination_source: if st.exited {
                st.termination_source.clone()
            } else {
                None
            },
        })
    }

    /// Paged full output: default the last `limit` lines; an explicit `offset` pages from the top
    /// (hermes `read_log`).
    pub fn read_log(&self, id: &str, offset: usize, limit: usize) -> Option<LogResult> {
        let session = self.get(id)?;
        self.reconcile(&session);
        let st = session.state.lock().unwrap();
        let full = strip_ansi(&st.output.to_lossy_string());
        let lines: Vec<&str> = full.lines().collect();
        let total_lines = lines.len();
        let selected: Vec<&str> = if offset == 0 && limit > 0 {
            lines.iter().rev().take(limit).rev().copied().collect()
        } else {
            lines.iter().skip(offset).take(limit).copied().collect()
        };
        if st.exited {
            self.consumed.lock().unwrap().insert(session.id.clone());
        }
        Some(LogResult {
            session_id: session.id.clone(),
            status: if st.exited { "exited" } else { "running" },
            showing: format!("{} lines", selected.len()),
            output: selected.join("\n"),
            total_lines,
        })
    }

    /// Block until the process exits, the wait window elapses, or the turn is cancelled. The
    /// waiter suppresses the duplicate completion notification for an exit it observes directly.
    pub async fn wait(
        &self,
        id: &str,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> Option<WaitResult> {
        let session = self.get(id)?;
        session.waiters.fetch_add(1, Ordering::SeqCst);
        let guard = WaiterGuard(&session.waiters);
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            // Arm the notification BEFORE checking state so an exit between the check and the
            // await cannot be missed.
            let notified = session.done.notified();
            self.reconcile(&session);
            {
                let st = session.state.lock().unwrap();
                if st.exited {
                    self.consumed.lock().unwrap().insert(session.id.clone());
                    drop(guard);
                    return Some(WaitResult {
                        status: WaitStatus::Exited,
                        exit_code: st.exit_code,
                        completion_reason: Some(st.reason.as_str()),
                        termination_source: st.termination_source.clone(),
                        output: strip_ansi(&st.output.tail_lossy(2000)),
                    });
                }
            }
            // Wake on exit / cancel / deadline — with a periodic tick so the orphaned-pipe
            // reconcile (whose grace expires between wakeups) still flips the wait promptly.
            let tick = tokio::time::Instant::now() + Duration::from_millis(RECONCILE_GRACE_MS);
            tokio::select! {
                _ = notified => {}
                _ = cancel.cancelled() => {
                    let st = session.state.lock().unwrap();
                    return Some(WaitResult {
                        status: WaitStatus::Interrupted,
                        exit_code: None,
                        completion_reason: None,
                        termination_source: None,
                        output: strip_ansi(&st.output.tail_lossy(1000)),
                    });
                }
                _ = tokio::time::sleep_until(deadline.min(tick)) => {
                    if tokio::time::Instant::now() >= deadline {
                        let st = session.state.lock().unwrap();
                        return Some(WaitResult {
                            status: WaitStatus::Timeout,
                            exit_code: None,
                            completion_reason: None,
                            termination_source: None,
                            output: strip_ansi(&st.output.tail_lossy(1000)),
                        });
                    }
                }
            }
        }
    }

    /// Terminate a process (SIGTERM to its process group, so the whole tree goes).
    pub fn kill(&self, id: &str, source: &str) -> KillResult {
        let Some(session) = self.get(id) else {
            return KillResult::NotFound;
        };
        if session.state.lock().unwrap().exited {
            let st = session.state.lock().unwrap();
            return KillResult::AlreadyExited {
                exit_code: st.exit_code,
            };
        }
        if let Err(e) = terminate_backend(&session) {
            return KillResult::Failed(e);
        }
        // Hermes records SIGTERM as exit -15 and the reason as killed; the reader thread's later
        // exited attempt no-ops (mark_exited is first-write-wins).
        self.mark_exited(
            &session,
            Some(-15),
            CompletionReason::Killed,
            Some(source.to_string()),
        );
        KillResult::Killed
    }

    /// Write raw bytes to a PTY session's stdin (no newline). Piped sessions have no stdin
    /// (spawned with stdin null — hermes parity).
    pub fn write_stdin(&self, id: &str, data: &str) -> StdinResult {
        self.stdin_op(id, |writer| {
            writer
                .write_all(data.as_bytes())
                .and_then(|()| writer.flush())
                .map(|()| data.len())
        })
    }

    /// Write `data` + newline (answer a prompt).
    pub fn submit_stdin(&self, id: &str, data: &str) -> StdinResult {
        self.stdin_op(id, |writer| {
            let line = format!("{data}\n");
            writer
                .write_all(line.as_bytes())
                .and_then(|()| writer.flush())
                .map(|()| line.len())
        })
    }

    /// Signal end-of-input: a PTY has no pipe EOF, so send EOT (^D) like hermes' `sendeof`.
    pub fn close_stdin(&self, id: &str) -> StdinResult {
        self.stdin_op(id, |writer| {
            writer
                .write_all(b"\x04")
                .and_then(|()| writer.flush())
                .map(|()| 0)
        })
    }

    fn stdin_op(
        &self,
        id: &str,
        op: impl FnOnce(&mut (dyn Write + Send)) -> std::io::Result<usize>,
    ) -> StdinResult {
        let Some(session) = self.get(id) else {
            return StdinResult::NotFound;
        };
        if session.state.lock().unwrap().exited {
            return StdinResult::AlreadyExited;
        }
        match &session.backend {
            Backend::Pty(pty) => {
                let mut writer = pty.writer.lock().unwrap();
                match writer.as_mut() {
                    Some(w) => match op(w.as_mut()) {
                        Ok(bytes) => StdinResult::Ok { bytes },
                        Err(e) => StdinResult::Failed(e.to_string()),
                    },
                    None => StdinResult::NotAvailable,
                }
            }
            Backend::Piped(_) | Backend::Detached => StdinResult::NotAvailable,
        }
    }

    /// All tracked processes (running + finished), optionally scoped to an owning session.
    pub fn list(&self, owner: Option<&SessionId>) -> Vec<ProcSummary> {
        let sessions: Vec<Arc<ProcSession>> = {
            let tracked = self.tracked.lock().unwrap();
            tracked
                .running
                .values()
                .chain(tracked.finished.values())
                .filter(|s| owner.is_none_or(|o| &s.owner == o))
                .cloned()
                .collect()
        };
        let now = self.clock.now_unix();
        let mut rows: Vec<ProcSummary> = sessions
            .iter()
            .map(|s| {
                let st = s.state.lock().unwrap();
                let mut command = s.command.clone();
                if command.len() > 200 {
                    let mut end = 200;
                    while end > 0 && !command.is_char_boundary(end) {
                        end -= 1;
                    }
                    command.truncate(end);
                }
                ProcSummary {
                    session_id: s.id.clone(),
                    command,
                    cwd: s.cwd.display().to_string(),
                    pid: s.pid,
                    started_unix: s.started_unix,
                    uptime_seconds: now.saturating_sub(s.started_unix),
                    status: if st.exited { "exited" } else { "running" },
                    output_preview: strip_ansi(&st.output.tail_lossy(200)),
                    exit_code: if st.exited { st.exit_code } else { None },
                }
            })
            .collect();
        rows.sort_by_key(|r| r.started_unix);
        rows
    }

    /// Currently-running process count (cheap; status surfaces).
    pub fn count_running(&self) -> usize {
        self.tracked.lock().unwrap().running.len()
    }

    /// Kill every running process (SIGTERM to each group) — the node shutdown path, so no
    /// process-group orphans outlive the daemon.
    pub fn shutdown(&self) {
        let running: Vec<String> = self
            .tracked
            .lock()
            .unwrap()
            .running
            .keys()
            .cloned()
            .collect();
        for id in running {
            let _ = self.kill(&id, "shutdown");
        }
    }

    /// TTL + LRU pruning (hermes `_prune_if_needed`; called under the tracked+consumed locks at
    /// spawn): expire finished sessions past the TTL (measured from start), then evict the oldest
    /// finished one when still at the tracked cap, then drop consumed markers for untracked ids.
    fn prune_locked(&self, tracked: &mut Tracked, consumed: &mut HashSet<String>) {
        let now = self.clock.now_unix();
        let ttl = self.cfg.finished_ttl_secs;
        let expired: Vec<String> = tracked
            .finished
            .iter()
            .filter(|(_, s)| now.saturating_sub(s.started_unix) > ttl)
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            tracked.finished.remove(&id);
            consumed.remove(&id);
        }
        let total = tracked.running.len() + tracked.finished.len();
        if total >= self.cfg.max_tracked && !tracked.finished.is_empty() {
            if let Some(oldest) = tracked
                .finished
                .iter()
                .min_by_key(|(_, s)| s.started_unix)
                .map(|(id, _)| id.clone())
            {
                tracked.finished.remove(&oldest);
                consumed.remove(&oldest);
            }
        }
        consumed.retain(|id| tracked.running.contains_key(id) || tracked.finished.contains_key(id));
    }

    /// Insert a synthetic finished session (no OS process) — TTL/LRU tests.
    #[cfg(test)]
    pub(crate) fn insert_finished_for_test(&self, id: &str, owner: &SessionId, started_unix: u64) {
        let session = Arc::new(ProcSession {
            id: id.to_string(),
            owner: owner.clone(),
            command: format!("synthetic {id}"),
            cwd: PathBuf::from("."),
            pid: None,
            started_unix,
            backend: Backend::Detached,
            state: Mutex::new(ProcState {
                exited: true,
                exit_code: Some(0),
                reason: CompletionReason::Exited,
                termination_source: None,
                output: RingBuffer::new(self.cfg.max_output_bytes),
                saw_output: true,
                watch: WatchState::default(),
                watch_patterns: Vec::new(),
                notify_on_complete: false,
            }),
            done: tokio::sync::Notify::new(),
            waiters: AtomicUsize::new(0),
            reconcile_armed_ms: Mutex::new(None),
        });
        self.tracked
            .lock()
            .unwrap()
            .finished
            .insert(id.to_string(), session);
    }

    /// Run one prune pass (tests).
    #[cfg(test)]
    pub(crate) fn prune_for_test(&self) {
        let mut tracked = self.tracked.lock().unwrap();
        let mut consumed = self.consumed.lock().unwrap();
        self.prune_locked(&mut tracked, &mut consumed);
    }

    /// Tracked ids, running + finished (tests).
    #[cfg(test)]
    pub(crate) fn tracked_ids_for_test(&self) -> Vec<String> {
        let tracked = self.tracked.lock().unwrap();
        tracked
            .running
            .keys()
            .chain(tracked.finished.keys())
            .cloned()
            .collect()
    }
}

/// A decrement-on-drop guard for the live-waiter count.
struct WaiterGuard<'a>(&'a AtomicUsize);

impl Drop for WaiterGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// SIGTERM the whole process group (POSIX), falling back to a direct child kill.
fn terminate_backend(session: &Arc<ProcSession>) -> Result<(), String> {
    match &session.backend {
        Backend::Piped(child) => {
            #[cfg(unix)]
            if let Some(pid) = session.pid {
                if nix::sys::signal::killpg(
                    nix::unistd::Pid::from_raw(pid as i32),
                    nix::sys::signal::Signal::SIGTERM,
                )
                .is_ok()
                {
                    return Ok(());
                }
            }
            child.kill().map_err(|e| e.to_string())
        }
        Backend::Pty(pty) => {
            #[cfg(unix)]
            if let Some(pid) = session.pid {
                // The pty child is its own session leader, so its pgid == its pid.
                if nix::sys::signal::killpg(
                    nix::unistd::Pid::from_raw(pid as i32),
                    nix::sys::signal::Signal::SIGTERM,
                )
                .is_ok()
                {
                    return Ok(());
                }
            }
            pty.child.lock().unwrap().kill().map_err(|e| e.to_string())
        }
        Backend::Detached => Err(
            "process has no runtime handle (recovered/synthetic session) and cannot be killed"
                .to_string(),
        ),
    }
}

/// Strip shell startup warnings from the head of the first output chunk (hermes
/// `_clean_shell_noise`).
fn clean_shell_noise(text: &str) -> String {
    let mut lines: Vec<&str> = text.split('\n').collect();
    while let Some(first) = lines.first() {
        if SHELL_NOISE.iter().any(|noise| first.contains(noise)) {
            lines.remove(0);
        } else {
            break;
        }
    }
    lines.join("\n")
}

/// The one-shot completion notification body (hermes `format_process_notification`).
fn format_completion(
    id: &str,
    command: &str,
    exit_code: Option<i32>,
    reason: CompletionReason,
    termination_source: Option<&str>,
    output_tail: &str,
) -> String {
    let signal = match exit_code {
        Some(-15) | Some(143) => ", SIGTERM",
        _ => "",
    };
    let status = match reason {
        CompletionReason::Killed => {
            format!("terminated by {}", termination_source.unwrap_or("daemon"))
        }
        CompletionReason::Exited if exit_code == Some(0) => "completed normally".to_string(),
        CompletionReason::Exited => "exited".to_string(),
    };
    let code = exit_code.map_or_else(|| "?".to_string(), |c| c.to_string());
    format!(
        "[IMPORTANT: Background process {id} {status} (exit code {code}{signal}).\n\
         Command: {command}\n\
         Output:\n{output_tail}]"
    )
}

/// A watch-pattern match notification body.
fn format_watch_match(
    id: &str,
    command: &str,
    pattern: &str,
    output: &str,
    suppressed: u32,
) -> String {
    let mut text = format!(
        "[IMPORTANT: Background process {id} matched watch pattern \"{pattern}\".\n\
         Command: {command}\n\
         Matched output:\n{output}"
    );
    if suppressed > 0 {
        text.push_str(&format!(
            "\n({suppressed} earlier matches were suppressed by rate limit)"
        ));
    }
    text.push(']');
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{FakeClock, RealClock};
    use async_trait::async_trait;

    /// A notifier capturing `(owner, text)` pairs for assertions.
    struct MockNotifier {
        seen: Mutex<Vec<(SessionId, String)>>,
        signal: tokio::sync::Notify,
    }

    impl MockNotifier {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                seen: Mutex::new(Vec::new()),
                signal: tokio::sync::Notify::new(),
            })
        }

        fn snapshot(&self) -> Vec<(SessionId, String)> {
            self.seen.lock().unwrap().clone()
        }

        /// Await until `pred` holds over the captured notifications (bounded; no fixed sleeps).
        async fn until(&self, pred: impl Fn(&[(SessionId, String)]) -> bool) {
            tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    if pred(&self.snapshot()) {
                        return;
                    }
                    self.signal.notified().await;
                }
            })
            .await
            .expect("notification did not arrive in time");
        }
    }

    #[async_trait]
    impl ProcessNotifier for MockNotifier {
        async fn notify(&self, owner: &SessionId, text: String) {
            self.seen.lock().unwrap().push((owner.clone(), text));
            self.signal.notify_waiters();
        }
    }

    fn registry() -> Arc<ProcessRegistry> {
        Arc::new(ProcessRegistry::new(
            RegistryConfig::default(),
            Arc::new(RealClock::new()),
        ))
    }

    fn req(owner: &str, line: &str) -> SpawnRequest {
        SpawnRequest {
            owner: SessionId::new(owner),
            line: line.to_string(),
            cwd: std::env::temp_dir(),
            pty: false,
            notify_on_complete: false,
            watch_patterns: Vec::new(),
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_wait_captures_output_and_exit_code() {
        let reg = registry();
        let session = reg
            .spawn(req("owner-a", "echo out-line; echo err-line 1>&2; exit 3"))
            .expect("spawn");
        assert!(session.id().starts_with("proc_"));
        assert_eq!(session.id().len(), "proc_".len() + 12, "proc_<12hex> ids");
        let result = reg
            .wait(
                session.id(),
                Duration::from_secs(10),
                &CancellationToken::new(),
            )
            .await
            .expect("tracked");
        assert_eq!(result.status, WaitStatus::Exited);
        assert_eq!(result.exit_code, Some(3));
        assert_eq!(result.completion_reason, Some("exited"));
        // stderr is merged into the one captured stream.
        assert!(result.output.contains("out-line"));
        assert!(result.output.contains("err-line"));
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kill_terminates_the_process_group() {
        let reg = registry();
        let session = reg.spawn(req("owner-a", "sleep 300")).expect("spawn");
        assert_eq!(reg.count_running(), 1);
        assert_eq!(reg.kill(session.id(), "process.kill"), KillResult::Killed);
        let polled = reg.poll(session.id()).expect("tracked");
        assert_eq!(polled.status, "exited");
        assert_eq!(polled.exit_code, Some(-15));
        assert_eq!(polled.completion_reason, Some("killed"));
        assert_eq!(polled.termination_source.as_deref(), Some("process.kill"));
        // Idempotent second kill reports already-exited.
        assert!(matches!(
            reg.kill(session.id(), "process.kill"),
            KillResult::AlreadyExited { .. }
        ));
        assert_eq!(reg.count_running(), 0);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn notify_on_complete_delivers_exactly_one_notification() {
        let reg = registry();
        let notifier = MockNotifier::new();
        reg.set_notifier(notifier.clone());
        let mut request = req("owner-n", "echo done-marker");
        request.notify_on_complete = true;
        let session = reg.spawn(request).expect("spawn");
        notifier.until(|seen| !seen.is_empty()).await;
        let seen = notifier.snapshot();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, SessionId::new("owner-n"));
        assert!(seen[0].1.contains("[IMPORTANT: Background process"));
        assert!(seen[0].1.contains(session.id()));
        assert!(seen[0].1.contains("completed normally"));
        assert!(seen[0].1.contains("done-marker"));
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_observing_the_exit_suppresses_the_notification() {
        let reg = registry();
        let notifier = MockNotifier::new();
        reg.set_notifier(notifier.clone());
        let mut request = req("owner-w", "sleep 0.2; echo fin");
        request.notify_on_complete = true;
        let session = reg.spawn(request).expect("spawn");
        let result = reg
            .wait(
                session.id(),
                Duration::from_secs(10),
                &CancellationToken::new(),
            )
            .await
            .expect("tracked");
        assert_eq!(result.status, WaitStatus::Exited);
        // The waiting agent already has the result — no duplicate notification turn.
        tokio::task::yield_now().await;
        assert!(
            notifier.snapshot().is_empty(),
            "a live waiter consumes the completion"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn watch_pattern_match_notifies_with_the_matched_lines() {
        let reg = registry();
        let notifier = MockNotifier::new();
        reg.set_notifier(notifier.clone());
        let mut request = req("owner-p", "echo starting; echo READY on :8080; sleep 60");
        request.watch_patterns = vec!["READY".to_string()];
        let session = reg.spawn(request).expect("spawn");
        notifier.until(|seen| !seen.is_empty()).await;
        let seen = notifier.snapshot();
        assert!(seen[0].1.contains("matched watch pattern \"READY\""));
        assert!(seen[0].1.contains("READY on :8080"));
        let _ = reg.kill(session.id(), "test-cleanup");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pty_session_answers_stdin_and_sees_the_echo() {
        let reg = registry();
        let mut request = req("owner-t", "read x; echo got:$x");
        request.pty = true;
        let session = match reg.spawn(request) {
            Ok(s) => s,
            // A sandbox without a usable /dev/ptmx cannot run this test; skip rather than flake.
            Err(SpawnError::Pty(e)) => {
                eprintln!("skipping pty test: {e}");
                return;
            }
            Err(e) => panic!("spawn: {e}"),
        };
        // Answer the prompt (submit appends the newline the `read` needs).
        let mut wrote = StdinResult::NotFound;
        for _ in 0..100 {
            wrote = reg.submit_stdin(session.id(), "hello-pty");
            if matches!(wrote, StdinResult::Ok { .. }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            matches!(wrote, StdinResult::Ok { .. }),
            "stdin write: {wrote:?}"
        );
        let result = reg
            .wait(
                session.id(),
                Duration::from_secs(10),
                &CancellationToken::new(),
            )
            .await
            .expect("tracked");
        assert_eq!(result.status, WaitStatus::Exited);
        assert!(
            result.output.contains("got:hello-pty"),
            "pty echo captured: {}",
            result.output
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn piped_sessions_have_no_stdin() {
        let reg = registry();
        let session = reg.spawn(req("owner-s", "sleep 60")).expect("spawn");
        assert_eq!(
            reg.write_stdin(session.id(), "x"),
            StdinResult::NotAvailable
        );
        assert_eq!(reg.close_stdin(session.id()), StdinResult::NotAvailable);
        let _ = reg.kill(session.id(), "test-cleanup");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_times_out_and_interrupts() {
        let reg = registry();
        let session = reg.spawn(req("owner-x", "sleep 300")).expect("spawn");
        let timed = reg
            .wait(
                session.id(),
                Duration::from_millis(50),
                &CancellationToken::new(),
            )
            .await
            .expect("tracked");
        assert_eq!(timed.status, WaitStatus::Timeout);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let interrupted = reg
            .wait(session.id(), Duration::from_secs(5), &cancel)
            .await
            .expect("tracked");
        assert_eq!(interrupted.status, WaitStatus::Interrupted);
        let _ = reg.kill(session.id(), "test-cleanup");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn log_pages_lines_and_list_scopes_by_owner() {
        let reg = registry();
        let session = reg
            .spawn(req("owner-l", "for i in 1 2 3 4 5; do echo line-$i; done"))
            .expect("spawn");
        reg.wait(
            session.id(),
            Duration::from_secs(10),
            &CancellationToken::new(),
        )
        .await
        .expect("tracked");
        let log = reg.read_log(session.id(), 0, 2).expect("tracked");
        assert_eq!(log.total_lines, 5);
        assert_eq!(log.output, "line-4\nline-5", "default = last N lines");
        let paged = reg.read_log(session.id(), 1, 2).expect("tracked");
        assert_eq!(paged.output, "line-2\nline-3", "offset pages from the top");

        assert_eq!(reg.list(Some(&SessionId::new("owner-l"))).len(), 1);
        assert!(reg.list(Some(&SessionId::new("someone-else"))).is_empty());
        assert_eq!(reg.list(None).len(), 1);
    }

    #[tokio::test]
    async fn ttl_prunes_finished_sessions_and_lru_evicts_oldest() {
        let clock = Arc::new(FakeClock::at(1_000_000));
        let cfg = RegistryConfig {
            max_tracked: 4,
            finished_ttl_secs: 1800,
            ..RegistryConfig::default()
        };
        let reg = Arc::new(ProcessRegistry::new(cfg, clock.clone()));
        let owner = SessionId::new("owner-ttl");
        reg.insert_finished_for_test("proc_aaaaaaaaaaaa", &owner, 1_000_000);
        reg.insert_finished_for_test("proc_bbbbbbbbbbbb", &owner, 1_000_100);
        // Within the TTL nothing expires.
        reg.prune_for_test();
        assert_eq!(reg.tracked_ids_for_test().len(), 2);
        // Past the TTL (measured from start), the older one expires.
        clock.advance_secs(1801);
        reg.prune_for_test();
        let ids = reg.tracked_ids_for_test();
        assert_eq!(ids, vec!["proc_bbbbbbbbbbbb".to_string()]);
        // LRU: at the cap, the oldest finished session is evicted (one per prune, hermes parity).
        reg.insert_finished_for_test("proc_cccccccccccc", &owner, 1_001_000);
        reg.insert_finished_for_test("proc_dddddddddddd", &owner, 1_001_001);
        reg.insert_finished_for_test("proc_eeeeeeeeeeee", &owner, 1_001_002);
        reg.prune_for_test();
        let mut ids = reg.tracked_ids_for_test();
        ids.sort();
        assert_eq!(
            ids,
            vec![
                "proc_cccccccccccc".to_string(),
                "proc_dddddddddddd".to_string(),
                "proc_eeeeeeeeeeee".to_string()
            ],
            "the oldest finished session was evicted at the cap"
        );
    }

    #[test]
    fn completion_formats_mirror_hermes() {
        let ok = format_completion(
            "proc_x",
            "make test",
            Some(0),
            CompletionReason::Exited,
            None,
            "tail",
        );
        assert!(ok.contains("completed normally (exit code 0)"));
        let killed = format_completion(
            "proc_x",
            "sleep 9",
            Some(-15),
            CompletionReason::Killed,
            Some("process.kill"),
            "",
        );
        assert!(killed.contains("terminated by process.kill (exit code -15, SIGTERM)"));
        let plain = format_completion("proc_x", "x", Some(2), CompletionReason::Exited, None, "");
        assert!(plain.contains("exited (exit code 2)"));
    }

    #[test]
    fn shell_noise_is_stripped_from_the_head_only() {
        let noisy =
            "bash: no job control in this shell\nreal output\nno job control in this shell kept";
        let cleaned = clean_shell_noise(noisy);
        assert!(cleaned.starts_with("real output"));
        assert!(cleaned.contains("kept"), "non-head noise lines are kept");
    }
}
