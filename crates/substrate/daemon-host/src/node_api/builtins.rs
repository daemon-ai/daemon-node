// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Built-in slash-command dispatch: the thin adapter that maps a resolved
//! [`Builtin`](crate::commands::Builtin) onto the node's existing typed ops (the cancel/model/mode/
//! approve logic lives once in the ops these helpers call, not here).

use super::*;

impl NodeApiImpl {
    /// Dispatch a resolved built-in command over the node's existing typed ops — a thin adapter, not
    /// a re-implementation (the logic for cancel/model/mode/approve lives once in the ops it calls).
    /// `command_invoke` has already gated access and verified a session is present for session-scoped
    /// commands. Each arm delegates to a per-builtin helper so this stays a flat dispatch.
    pub(crate) async fn run_builtin(
        &self,
        builtin: crate::commands::Builtin,
        invocation: &CommandInvocation,
    ) -> Result<CommandOutput, ApiError> {
        use crate::commands::Builtin;
        match builtin {
            Builtin::Help => Ok(self.builtin_help()),
            Builtin::Whoami => Ok(self.builtin_whoami()),
            Builtin::Version => Ok(builtin_version()),
            Builtin::Status => Ok(self.builtin_status().await),
            Builtin::Usage => Ok(self.builtin_usage().await),
            Builtin::Sessions => Ok(self.builtin_sessions().await),
            Builtin::Stop => self.builtin_stop(invocation).await,
            Builtin::Model => self.builtin_model(invocation).await,
            Builtin::Mode => self.builtin_mode(invocation).await,
            Builtin::Title => self.builtin_title(invocation).await,
            Builtin::Approve | Builtin::Deny => {
                self.builtin_approve_deny(builtin, invocation).await
            }
        }
    }

    /// `/help` — the catalog's advertised specs, rendered.
    fn builtin_help(&self) -> CommandOutput {
        let specs = self
            .commands
            .load()
            .as_ref()
            .map(|r| r.specs())
            .unwrap_or_default();
        CommandOutput {
            text: crate::commands::render_help(&specs),
            ephemeral: true,
        }
    }

    /// `/whoami` — the node's active default profile + partition.
    fn builtin_whoami(&self) -> CommandOutput {
        CommandOutput {
            text: format!(
                "profile: {}\npartition: {:?}",
                self.default_local_profile, self.partition
            ),
            ephemeral: true,
        }
    }

    /// `/status` — the telemetry health/session/job summary.
    async fn builtin_status(&self) -> CommandOutput {
        let t = self.telemetry().await;
        CommandOutput {
            text: format!(
                "healthy: {}\nsessions: {} ({} active)\npending jobs: {}, wakes: {}\nevents: {}",
                t.healthy, t.sessions, t.active, t.pending_jobs, t.pending_wakes, t.events
            ),
            ephemeral: true,
        }
    }

    /// `/usage` — the folded token/cache/cost accounting line.
    async fn builtin_usage(&self) -> CommandOutput {
        let t = self.telemetry().await;
        let u = &t.usage;
        CommandOutput {
            text: format!(
                "tokens in/out: {}/{}\ncache read/write: {}/{}\nreasoning: {}\nest. cost: ${:.4}",
                u.input_tokens,
                u.output_tokens,
                u.cache_read_tokens,
                u.cache_write_tokens,
                u.reasoning_tokens,
                u.cost_micros as f64 / 1_000_000.0
            ),
            ephemeral: true,
        }
    }

    /// `/sessions` — the roster as a `id — title` listing.
    async fn builtin_sessions(&self) -> CommandOutput {
        let roster = self.sessions().await;
        let mut text = format!("{} session(s):\n", roster.len());
        for s in &roster {
            let title = s.title.as_deref().unwrap_or("(untitled)");
            text.push_str(&format!("  {} — {}\n", s.session.as_str(), title));
        }
        CommandOutput {
            text,
            ephemeral: true,
        }
    }

    /// `/stop` — cancel in-flight work on the invoking session.
    async fn builtin_stop(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandOutput, ApiError> {
        let session = require_session(invocation, "stop")?;
        self.cancel(session).await?;
        Ok(CommandOutput {
            text: "cancelled in-flight work".into(),
            ephemeral: true,
        })
    }

    /// `/model <model-id>` — switch the invoking session's model.
    async fn builtin_model(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandOutput, ApiError> {
        let args = invocation.args.trim();
        if args.is_empty() {
            return Err(ApiError::Other("usage: /model <model-id>".into()));
        }
        let session = require_session(invocation, "model")?;
        self.set_session_model(session, args.to_string(), None)
            .await?;
        Ok(CommandOutput {
            text: format!("session model set to {args}"),
            ephemeral: true,
        })
    }

    /// `/mode` (and the `/yolo`,`/fast` aliases) — switch the invoking session's approval mode.
    async fn builtin_mode(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandOutput, ApiError> {
        let args = invocation.args.trim();
        let session = require_session(invocation, "mode")?;
        let mode = resolve_approval_mode(&invocation.name, args)?;
        self.set_session_mode(session, mode).await?;
        Ok(CommandOutput {
            text: format!("approval mode set to {mode:?}"),
            ephemeral: true,
        })
    }

    /// `/title <new title>` — rename the invoking session.
    async fn builtin_title(
        &self,
        invocation: &CommandInvocation,
    ) -> Result<CommandOutput, ApiError> {
        let args = invocation.args.trim();
        if args.is_empty() {
            return Err(ApiError::Other("usage: /title <new title>".into()));
        }
        let session = require_session(invocation, "title")?;
        let patch = SessionMetaPatch {
            title: Some(Some(args.to_string())),
            ..SessionMetaPatch::default()
        };
        self.session_update_meta(session, patch).await?;
        Ok(CommandOutput {
            text: format!("title set to {args:?}"),
            ephemeral: true,
        })
    }

    /// `/approve` / `/deny <request-id>` — resolve a pending approval on the invoking session.
    async fn builtin_approve_deny(
        &self,
        builtin: crate::commands::Builtin,
        invocation: &CommandInvocation,
    ) -> Result<CommandOutput, ApiError> {
        let allow = matches!(builtin, crate::commands::Builtin::Approve);
        let args = invocation.args.trim();
        if args.is_empty() {
            return Err(ApiError::Other(format!(
                "usage: /{} <request-id>",
                if allow { "approve" } else { "deny" }
            )));
        }
        let session = require_session(invocation, "approval")?;
        self.approval_decide(session, args.to_string(), allow)
            .await?;
        Ok(CommandOutput {
            text: format!(
                "request {args} {}",
                if allow { "approved" } else { "denied" }
            ),
            ephemeral: true,
        })
    }
}

/// The invoking session, or a `"<what> requires a session"` error — the shared guard the
/// session-scoped builtins (`/stop`, `/model`, `/mode`, `/title`, `/approve`, `/deny`) front with.
fn require_session(invocation: &CommandInvocation, what: &str) -> Result<SessionId, ApiError> {
    invocation
        .session
        .clone()
        .ok_or_else(|| ApiError::Other(format!("{what} requires a session")))
}

/// `/version` — the daemon crate version (no node state needed).
fn builtin_version() -> CommandOutput {
    CommandOutput {
        text: format!("daemon {}", env!("CARGO_PKG_VERSION")),
        ephemeral: true,
    }
}

/// Resolve the requested [`ApprovalMode`] from a `/mode` invocation: the `yolo`/`fast` aliases map
/// directly, otherwise the first argument is parsed (`yolo`/`auto`, `fast`/`accept`, `ask`, `deny`).
fn resolve_approval_mode(name: &str, args: &str) -> Result<ApprovalMode, ApiError> {
    let key = match name
        .trim()
        .trim_start_matches('/')
        .to_ascii_lowercase()
        .as_str()
    {
        "yolo" => "yolo".to_string(),
        "fast" => "fast".to_string(),
        _ => args
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase(),
    };
    match key.as_str() {
        "yolo" | "auto" | "autoallow" | "auto-allow" => Ok(ApprovalMode::AutoAllow),
        "fast" | "accept" | "acceptedits" | "accept-edits" => Ok(ApprovalMode::AcceptEdits),
        "ask" | "default" => Ok(ApprovalMode::Ask),
        "deny" | "reject" => Ok(ApprovalMode::Deny),
        "" => Err(ApiError::Other("usage: /mode <yolo|fast|ask|deny>".into())),
        other => Err(ApiError::Other(format!("unknown approval mode: {other}"))),
    }
}

/// Map a [`daemon_core::CommandError`] to the wire [`ApiError`] at the command boundary.
pub(crate) fn command_err_to_api(err: daemon_core::CommandError) -> ApiError {
    use daemon_core::CommandError::*;
    match err {
        Unknown(name) => ApiError::Other(format!("unknown command: {name}")),
        BadArgs(msg) => ApiError::Other(format!("invalid arguments: {msg}")),
        MissingSession => ApiError::Other("command requires an active session".into()),
        Failed(msg) => ApiError::Other(msg),
    }
}
