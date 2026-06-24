//! The command-provider seam — daemon-authoritative operator/user commands.
//!
//! Commands are the human-invoked counterpart to the model-facing §12 [`ToolRegistry`](crate::tools):
//! a thin client (GUI/TUI) enumerates the catalog and dispatches an invocation, and the daemon owns
//! both the declarative catalog and the handlers. This module is the *engine-side* half — the
//! [`CommandProvider`] trait a subsystem (the §10 [`ContextEngine`](crate::context::ContextEngine),
//! a §11 [`MemoryProvider`](crate::memory::MemoryProvider), or a plugin) implements to contribute
//! commands, plus the core-local [`CommandSpec`]/[`CommandInvocation`]/[`CommandOutput`] vocabulary.
//!
//! These mirror the `daemon-api` wire DTOs of the same name; the node layer maps between the two at
//! the contract boundary, exactly as it does for every other engine type (the engine crate stays
//! free of the wire/contract layer). It is deliberately a *separate* seam from [`ToolRegistry`]:
//! tools are what the model calls mid-turn, commands are what an operator invokes out-of-band.
//!
//! Like tools, a provider advertises **metadata** ([`commands`](CommandProvider::commands)) decoupled
//! from the **handler** ([`run_command`](CommandProvider::run_command)); the node-side registry is
//! the single live catalog that unifies built-in node ops with every provider's contribution.

use crate::conversation::Conversation;
use async_trait::async_trait;
use daemon_common::SessionId;
use std::sync::Arc;

/// The minimum access tier required to run a command — the `slash_access.py` analog. The node's
/// access policy compares an invocation's resolved tier against this floor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CommandAccess {
    /// Any authenticated caller may run it (the read-only floor: e.g. `help`/`whoami`/`status`).
    #[default]
    User,
    /// Operator/admin only — mutating or node-wide ops.
    Admin,
}

/// Whether a command applies to a specific session or the node as a whole.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CommandScope {
    /// Operates on a specific session (the invocation must carry a [`SessionId`]).
    #[default]
    Session,
    /// Operates on the node as a whole (no session required).
    Node,
}

/// Declarative command metadata a provider advertises (the `CommandDef` analog, core-local). The
/// node maps this to the `daemon-api` wire `CommandSpec`, filling `source` from the provider name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandSpec {
    /// Canonical name without the leading slash (e.g. `"lcm"`, `"memory"`).
    pub name: String,
    /// Alternative names that resolve to the same command.
    pub aliases: Vec<String>,
    /// One-line human description.
    pub summary: String,
    /// Grouping for client menus (e.g. `"Context"`, `"Memory"`).
    pub category: String,
    /// Short argument placeholder hint (e.g. `"<subcommand>"`).
    pub args_hint: String,
    /// Tab-completable subcommands (e.g. `["status", "doctor", "backup"]`).
    pub subcommands: Vec<String>,
    /// Whether the command applies to a session or the whole node.
    pub scope: CommandScope,
    /// Whether the command mutates durable state (clients treat it as non-idempotent / confirm).
    pub side_effecting: bool,
    /// A UX hint that the client should confirm before running (destructive `apply` variants).
    pub confirm: bool,
    /// The minimum access tier required to run it.
    pub min_access: CommandAccess,
}

impl CommandSpec {
    /// A minimal read-only, user-tier, session-scoped spec — the common starting point a builder
    /// refines. (`new("lcm").summary(...).subcommand("status")...`)
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Set the one-line summary.
    pub fn summary(mut self, s: impl Into<String>) -> Self {
        self.summary = s.into();
        self
    }

    /// Set the menu category.
    pub fn category(mut self, c: impl Into<String>) -> Self {
        self.category = c.into();
        self
    }

    /// Set the argument hint.
    pub fn args_hint(mut self, h: impl Into<String>) -> Self {
        self.args_hint = h.into();
        self
    }

    /// Append an alias.
    pub fn alias(mut self, a: impl Into<String>) -> Self {
        self.aliases.push(a.into());
        self
    }

    /// Replace the subcommand list.
    pub fn subcommands<I, S>(mut self, subs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.subcommands = subs.into_iter().map(Into::into).collect();
        self
    }

    /// Mark the command node-scoped (no session required).
    pub fn node_scoped(mut self) -> Self {
        self.scope = CommandScope::Node;
        self
    }

    /// Mark the command mutating + confirm-on-run and admin-only (the destructive/`apply` variants).
    pub fn mutating(mut self) -> Self {
        self.side_effecting = true;
        self.confirm = true;
        self.min_access = CommandAccess::Admin;
        self
    }

    /// Require the admin tier (without the mutating/confirm flags).
    pub fn admin(mut self) -> Self {
        self.min_access = CommandAccess::Admin;
        self
    }
}

/// A request to run a command. `args` is the raw trailing argument string, parsed by the handler
/// (mirrors hermes' `fn(raw_args: str)`); `session` is set for [`CommandScope::Session`] commands.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandInvocation {
    /// The command name or alias (the registry resolves it; with or without a leading slash).
    pub name: String,
    /// The raw trailing argument string (may be empty).
    pub args: String,
    /// The target session, for session-scoped commands.
    pub session: Option<SessionId>,
}

impl CommandInvocation {
    /// Split [`args`](Self::args) into whitespace-separated tokens (the cheap parse most handlers use).
    pub fn tokens(&self) -> Vec<&str> {
        self.args.split_whitespace().collect()
    }

    /// The first whitespace-separated token of [`args`](Self::args), if any — the subcommand verb.
    pub fn subcommand(&self) -> Option<&str> {
        self.args.split_whitespace().next()
    }

    /// The argument string with the leading subcommand verb stripped (the rest of the line).
    pub fn rest(&self) -> &str {
        match self.args.split_once(char::is_whitespace) {
            Some((_, rest)) => rest.trim_start(),
            None => "",
        }
    }
}

/// The rendered result of a command invocation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandOutput {
    /// The rendered output text the client displays.
    pub text: String,
    /// When `true`, client-local feedback that must not enter the transcript/journal.
    pub ephemeral: bool,
}

impl CommandOutput {
    /// A plain text result (non-ephemeral).
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ephemeral: false,
        }
    }

    /// An ephemeral (client-local) text result.
    pub fn ephemeral(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ephemeral: true,
        }
    }
}

/// A command failure (mapped to the wire `ApiError` at the node boundary).
#[derive(Debug, thiserror::Error)]
pub enum CommandError {
    /// The command (or subcommand) is not known to this provider.
    #[error("unknown command: {0}")]
    Unknown(String),
    /// The arguments were malformed / a required argument was missing.
    #[error("invalid arguments: {0}")]
    BadArgs(String),
    /// A session-scoped command was invoked without a session.
    #[error("command requires an active session")]
    MissingSession,
    /// The handler ran but failed.
    #[error("{0}")]
    Failed(String),
}

/// The context handed to a [`CommandProvider::run_command`] handler: the resolved session (for a
/// session-scoped command) and, when the node can supply it, a read-only view of that session's
/// conversation. Mirrors [`ToolCx`](crate::turn::TurnCx) for the command surface — a provider reads
/// what it needs and returns a [`CommandOutput`]; it does not drive the turn loop.
pub struct CommandCx<'a> {
    /// The target session id (for session-scoped commands).
    pub session: Option<SessionId>,
    /// A read-only view of the session conversation, when the node has a live one to lend.
    pub conversation: Option<&'a Conversation>,
}

impl<'a> CommandCx<'a> {
    /// A node-scoped context (no session, no conversation).
    pub fn node() -> Self {
        Self {
            session: None,
            conversation: None,
        }
    }

    /// A session-scoped context with no conversation view.
    pub fn session(session: SessionId) -> Self {
        Self {
            session: Some(session),
            conversation: None,
        }
    }

    /// Attach a read-only conversation view.
    pub fn with_conversation(mut self, conv: &'a Conversation) -> Self {
        self.conversation = Some(conv);
        self
    }
}

/// A subsystem that contributes operator/user commands — the `register_command` /
/// `register_context_engine` analog. A [`ContextEngine`](crate::context::ContextEngine) or
/// [`MemoryProvider`](crate::memory::MemoryProvider) implements it (and exposes itself via
/// [`ContextEngine::command_provider`](crate::context::ContextEngine::command_provider) /
/// [`MemoryProvider::command_provider`](crate::memory::MemoryProvider::command_provider)), as may a
/// plugin. The node-side registry merges every provider's [`commands`](Self::commands) into one live
/// catalog and routes an invocation to the owning provider's [`run_command`](Self::run_command).
#[async_trait]
pub trait CommandProvider: Send + Sync {
    /// A stable label for diagnostics + as the catalog `source` (e.g. `"lcm"`, `"mnemosyne"`).
    fn name(&self) -> &str;

    /// The commands this provider advertises (metadata only; decoupled from the handler).
    fn commands(&self) -> Vec<CommandSpec>;

    /// Run a previously-advertised command. The registry has already resolved the name/alias and
    /// gated access; the provider parses [`CommandInvocation::args`] and returns rendered output.
    async fn run_command(
        &self,
        invocation: &CommandInvocation,
        cx: &CommandCx<'_>,
    ) -> Result<CommandOutput, CommandError>;
}

/// Convenience: a no-op provider with no commands (the default a subsystem inherits before it opts
/// in). Mostly useful in tests and as documentation of the empty case.
pub struct NoCommands;

#[async_trait]
impl CommandProvider for NoCommands {
    fn name(&self) -> &str {
        "none"
    }
    fn commands(&self) -> Vec<CommandSpec> {
        Vec::new()
    }
    async fn run_command(
        &self,
        invocation: &CommandInvocation,
        _cx: &CommandCx<'_>,
    ) -> Result<CommandOutput, CommandError> {
        Err(CommandError::Unknown(invocation.name.clone()))
    }
}

/// A boxed command provider handle (the shape the registry and [`EngineProfile`](crate::profile::EngineProfile)
/// hold).
pub type CommandProviderHandle = Arc<dyn CommandProvider>;
