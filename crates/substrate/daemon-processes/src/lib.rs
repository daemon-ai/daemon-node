// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-processes` — the host-owned background-process registry (hermes `process_registry.py`).
//!
//! The engine's [`ExecutionEnvironment`] runs only **transient** commands (exec/mod.rs §13: "a
//! long-running watched OS process is host-owned"), so background processes live here: a resident
//! node service, constructed once at assembly and shared by the `shell` tool (spawn), the `process`
//! tool (poll/log/wait/kill/stdin), and the exit notifier. It never lives on a `TurnCx` — background
//! processes outlive the turn that spawned them, and a turn cancel must not kill them.
//!
//! Ports the hermes behaviors + limits: a rolling per-process output ring (200 KB), an LRU-bounded
//! tracked set (64) with a finished-TTL (30 min), stdin write/submit/close (PTY sessions), PTY
//! spawn (30×120), watch-patterns with the per-session rate limit (1 match / 15 s, 3 strike windows
//! → disabled + promoted to notify-on-complete) and the cross-session global circuit breaker
//! (15 / 10 s → 30 s cooldown). Completion/watch notifications are formatted here
//! (`[IMPORTANT: ...]`, hermes `format_process_notification`) and delivered through the injected
//! [`ProcessNotifier`] — the node adapter routes them into the owning session (a live `StartTurn`
//! or the durable pending-input seam).
//!
//! [`ExecutionEnvironment`]: https://docs.rs/daemon-core

#![forbid(unsafe_code)]

mod ansi;
mod clock;
mod registry;
mod ring;
mod watch;

use std::path::PathBuf;

use async_trait::async_trait;
use daemon_common::SessionId;
use serde::{Deserialize, Serialize};

pub use ansi::strip_ansi;
pub use clock::{Clock, FakeClock, RealClock};
pub use registry::{
    KillResult, LogResult, PollResult, ProcSummary, ProcessRegistry, SpawnError, SpawnRequest,
    StdinResult, WaitResult, WaitStatus,
};

/// Where completion / watch notifications go: the node implements this over its session surface
/// (`NodeApiImpl::inject_session_input`), late-bound after assembly like cron's delivery handle.
/// `text` is the fully-formatted `[IMPORTANT: ...]` message; `owner` is the session that spawned
/// the process.
#[async_trait]
pub trait ProcessNotifier: Send + Sync {
    /// Deliver one notification into `owner`'s conversation (driving a reactive turn).
    async fn notify(&self, owner: &SessionId, text: String);
}

/// `[processes]` — the registry-level knobs (hermes `process_registry.py` module constants).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RegistryConfig {
    /// Rolling per-process output buffer cap in bytes (hermes `MAX_OUTPUT_CHARS`).
    pub max_output_bytes: usize,
    /// Max concurrently tracked processes, running + finished (hermes `MAX_PROCESSES`; LRU-pruned).
    pub max_tracked: usize,
    /// How long a finished process stays queryable, measured from its **start** (hermes
    /// `FINISHED_TTL_SECONDS` prunes on `now - started_at`).
    pub finished_ttl_secs: u64,
    /// Reader chunk size in bytes.
    pub reader_chunk_bytes: usize,
    /// PTY dimensions (rows × cols) for `pty=true` spawns.
    pub pty_rows: u16,
    /// PTY columns.
    pub pty_cols: u16,
    /// Per-session watch rate limit: minimum spacing between two emitted matches (hermes
    /// `WATCH_MIN_INTERVAL_SECONDS`).
    pub watch_min_interval_secs: u64,
    /// Consecutive strike windows before watch is disabled + promoted to notify-on-complete
    /// (hermes `WATCH_STRIKE_LIMIT`).
    pub watch_strike_limit: u32,
    /// Global breaker: max admitted watch matches per window, across all sessions (hermes
    /// `WATCH_GLOBAL_MAX_PER_WINDOW`).
    pub watch_global_max_per_window: u32,
    /// Global breaker window length (hermes `WATCH_GLOBAL_WINDOW_SECONDS`).
    pub watch_global_window_secs: u64,
    /// Global breaker cooldown once tripped (hermes `WATCH_GLOBAL_COOLDOWN_SECONDS`).
    pub watch_global_cooldown_secs: u64,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_output_bytes: 200_000,
            max_tracked: 64,
            finished_ttl_secs: 1800,
            reader_chunk_bytes: 4096,
            pty_rows: 30,
            pty_cols: 120,
            watch_min_interval_secs: 15,
            watch_strike_limit: 3,
            watch_global_max_per_window: 15,
            watch_global_window_secs: 10,
            watch_global_cooldown_secs: 30,
        }
    }
}

/// `[shell]` — the shell-tool knobs (hermes `terminal_tool.py` timeouts + truncation).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShellConfig {
    /// Default foreground timeout in seconds (hermes `TERMINAL_TIMEOUT`).
    pub timeout_default_secs: u64,
    /// Hard foreground timeout cap: a larger request is rejected with a "use background=true"
    /// nudge (hermes `FOREGROUND_MAX_TIMEOUT`).
    pub timeout_max_secs: u64,
    /// Foreground output cap in bytes before head/tail truncation applies.
    pub truncate_max_bytes: usize,
    /// Head share of the truncation budget in percent (hermes keeps 40% head + 60% tail).
    pub truncate_head_pct: u8,
    /// Whether the per-session working directory persists across `shell` calls (`cd` semantics).
    pub persist_cwd: bool,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            timeout_default_secs: 180,
            timeout_max_secs: 600,
            truncate_max_bytes: 200_000,
            truncate_head_pct: 40,
            persist_cwd: true,
        }
    }
}

/// The combined process-service policy carried through `NodeAssembly`: the registry knobs plus the
/// shell-tool knobs (the binary maps its `[processes]` / `[shell]` TOML sections onto this).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessesConfig {
    /// Registry-level limits (`[processes]`).
    pub registry: RegistryConfig,
    /// Shell-tool limits (`[shell]`).
    pub shell: ShellConfig,
}

/// Truncate `s` to at most `max_bytes`, keeping `head_pct`% of the budget from the head and the
/// rest from the tail (hermes keeps 40% head — errors surface early — and 60% tail — the most
/// recent output), splicing an omission notice between. Char-boundary safe.
pub fn truncate_head_tail(s: &str, max_bytes: usize, head_pct: u8) -> String {
    if s.len() <= max_bytes || max_bytes == 0 {
        return s.to_string();
    }
    let head_budget = max_bytes.saturating_mul(usize::from(head_pct.min(100))) / 100;
    let tail_budget = max_bytes - head_budget;
    let mut head_end = head_budget.min(s.len());
    while head_end > 0 && !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = s.len().saturating_sub(tail_budget);
    while tail_start < s.len() && !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let omitted = tail_start.saturating_sub(head_end);
    format!(
        "{}\n\n... [{} bytes truncated] ...\n\n{}",
        &s[..head_end],
        omitted,
        &s[tail_start..]
    )
}

/// A validated working directory for a spawn (the shell tool resolves + contains it first).
pub type WorkDir = PathBuf;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_head_and_tail_shares() {
        let s = "a".repeat(50) + &"b".repeat(1000) + &"c".repeat(60);
        let out = truncate_head_tail(&s, 100, 40);
        assert!(out.starts_with(&"a".repeat(40)), "40% head kept");
        assert!(out.ends_with(&"c".repeat(60)), "60% tail kept");
        assert!(out.contains("bytes truncated"));
    }

    #[test]
    fn truncate_is_noop_under_cap_and_char_safe() {
        assert_eq!(truncate_head_tail("short", 100, 40), "short");
        // Multi-byte chars at both split points must not panic.
        let s = "é".repeat(200);
        let out = truncate_head_tail(&s, 100, 40);
        assert!(out.contains("bytes truncated"));
    }
}
