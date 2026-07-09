// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The node-side command registry — the single live catalog behind the daemon-authoritative
//! command surface ([`ControlApi::command_list`](daemon_api::ControlApi::command_list) /
//! [`command_invoke`](daemon_api::ControlApi::command_invoke)).
//!
//! It unifies three command sources into one alias-aware catalog:
//! 1. **built-in node ops** — thin [`CommandSpec`]s over the existing typed `NodeApi` ops (the
//!    handler is [`NodeApiImpl`](crate::node_api::NodeApiImpl)'s own dispatch; no op logic is
//!    re-implemented),
//! 2. **provider-contributed** — every [`daemon_core::CommandProvider`] the engine profile yields
//!    (the §10 context engine's `/lcm`/`/compress`, the §11 memory provider's `/memory`),
//! 3. **plugin-contributed** — registered through the same provider seam.
//!
//! Resolution is alias-aware (a name or any alias, with or without a leading `/`); a later
//! registration that collides with an existing name/alias is **rejected** (first wins), so the
//! catalog stays unambiguous. Access is gated by [`CommandSpec::min_access`] against the caller's
//! tier ([`caller_access`]), with the read-only `User` floor always allowed — the `slash_access.py`
//! analog folded into the node, not a parallel permission system.

use daemon_api::{CommandAccess, CommandScope, CommandSpec};
use daemon_core::{
    CommandAccess as CoreAccess, CommandProviderHandle, CommandScope as CoreScope,
    CommandSpec as CoreSpec,
};
use daemon_protocol::Origin;
use std::collections::HashMap;

/// A built-in command's canonical identity — the node dispatches each directly over its own typed
/// `NodeApi` ops (see `NodeApiImpl::run_builtin`). The catalog `CommandSpec` is the advertised
/// metadata; this enum is the resolved handler tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    /// List the command catalog.
    Help,
    /// Node + caller identity (profile / partition).
    Whoami,
    /// The daemon build version.
    Version,
    /// A node/session status line (telemetry projection).
    Status,
    /// Folded usage + cost (telemetry).
    Usage,
    /// The session roster.
    Sessions,
    /// Cancel in-flight work for the session.
    Stop,
    /// Set the session model.
    Model,
    /// Set the session approval mode (`yolo` => auto-allow, `fast` => accept-edits).
    Mode,
    /// Rename the session (title).
    Title,
    /// Approve a parked edit-approval request (`approve <request_id>`).
    Approve,
    /// Deny a parked edit-approval request (`deny <request_id>`).
    Deny,
}

/// Who owns a resolved command's handler.
pub enum Owner {
    /// A built-in node op, dispatched directly by [`NodeApiImpl`](crate::node_api::NodeApiImpl).
    Builtin(Builtin),
    /// A subsystem-contributed command, routed to its [`daemon_core::CommandProvider`].
    Provider(CommandProviderHandle),
}

/// One catalog entry: the advertised wire spec plus the resolved handler owner.
pub struct Entry {
    /// The advertised metadata the client renders + autocompletes from.
    pub spec: CommandSpec,
    /// The handler owner the node routes an invocation to.
    pub owner: Owner,
}

/// The unified, alias-aware command catalog.
#[derive(Default)]
pub struct CommandRegistry {
    entries: Vec<Entry>,
    /// Lowercased name/alias -> entry index. Conflicts are rejected at insert time.
    index: HashMap<String, usize>,
}

impl CommandRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry seeded with the built-in node-op commands (the catalog every node exposes before
    /// any provider is folded in).
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        for (spec, builtin) in builtin_catalog() {
            reg.insert(Entry {
                spec,
                owner: Owner::Builtin(builtin),
            });
        }
        reg
    }

    /// Fold a [`daemon_core::CommandProvider`]'s commands into the catalog, mapping each core
    /// [`CoreSpec`] to the wire [`CommandSpec`] (with `source` set to the provider name). A command
    /// whose name or any alias collides with an existing entry is rejected (logged + skipped).
    pub fn register_provider(&mut self, provider: CommandProviderHandle) {
        let source = provider.name().to_string();
        for core in provider.commands() {
            let spec = to_wire_spec(&core, &source);
            self.insert(Entry {
                spec,
                owner: Owner::Provider(provider.clone()),
            });
        }
    }

    /// Fold every provider in `providers` into the catalog (convenience over
    /// [`register_provider`](Self::register_provider)).
    pub fn register_providers(
        &mut self,
        providers: impl IntoIterator<Item = CommandProviderHandle>,
    ) {
        for p in providers {
            self.register_provider(p);
        }
    }

    /// Insert an entry, indexing its name + aliases. A name/alias already in the index is a
    /// conflict: the colliding key is left pointing at the first registrant and a warning is logged
    /// (the catalog stays unambiguous).
    fn insert(&mut self, entry: Entry) {
        let idx = self.entries.len();
        let mut keys = Vec::with_capacity(entry.spec.aliases.len() + 1);
        keys.push(normalize(&entry.spec.name));
        for a in &entry.spec.aliases {
            keys.push(normalize(a));
        }
        // Reject the whole entry if any of its keys already resolves (first registration wins).
        if let Some(conflict) = keys.iter().find(|k| self.index.contains_key(*k)) {
            tracing::warn!(
                command = %entry.spec.name,
                source = %entry.spec.source,
                conflict = %conflict,
                "command name/alias already registered; rejecting duplicate"
            );
            return;
        }
        for k in keys {
            self.index.insert(k, idx);
        }
        self.entries.push(entry);
    }

    /// Resolve a name or alias (leading `/` and case are ignored) to its entry.
    pub fn resolve(&self, name: &str) -> Option<&Entry> {
        self.index.get(&normalize(name)).map(|i| &self.entries[*i])
    }

    /// The full catalog as advertised wire specs (the `command_list` payload).
    pub fn specs(&self) -> Vec<CommandSpec> {
        self.entries.iter().map(|e| e.spec.clone()).collect()
    }

    /// The number of distinct commands registered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Normalize a command token for resolution: strip a leading `/`, lowercase, trim.
fn normalize(name: &str) -> String {
    name.trim().trim_start_matches('/').to_ascii_lowercase()
}

/// Map a core [`CoreSpec`] to the wire [`CommandSpec`], stamping `source` with the provider name.
fn to_wire_spec(core: &CoreSpec, source: &str) -> CommandSpec {
    CommandSpec {
        name: core.name.clone(),
        aliases: core.aliases.clone(),
        summary: core.summary.clone(),
        category: core.category.clone(),
        args_hint: core.args_hint.clone(),
        subcommands: core.subcommands.clone(),
        scope: match core.scope {
            CoreScope::Node => CommandScope::Node,
            CoreScope::Session => CommandScope::Session,
        },
        surfaces: Vec::new(),
        side_effecting: core.side_effecting,
        confirm: core.confirm,
        min_access: match core.min_access {
            CoreAccess::Admin => CommandAccess::Admin,
            CoreAccess::User => CommandAccess::User,
        },
        source: source.to_string(),
    }
}

/// The caller's access tier for command gating — the `slash_access.py` operator-vs-user axis, now
/// derived from the **authenticated principal** (the authz-core request context), not from
/// origin-presence. A principal holding an operator/admin capability (`ControlWrite` or
/// `AccessAdmin`) gets [`CommandAccess::Admin`]; everyone else — including an unauthenticated /
/// empty context — gets the read-only [`CommandAccess::User`] floor. The read-only floor is always
/// allowed (see [`access_allows`]), so a chat user can still run `help`/`status` but not the
/// admin-tier mutating/node-wide ops.
///
/// Fail-closed inversion (authz core): the absence of identity no longer implies admin (was
/// `None => Admin`). The trusted in-process / FFI / local-Unix paths keep the admin tier because
/// they run inside a [`RequestContext::system()`](crate::request_context::RequestContext::system)
/// scope, whose full-capability principal satisfies the check. `origin` is retained for signature
/// stability and future per-origin policy, but no longer grants a tier.
pub fn caller_access(_origin: Option<&Origin>) -> CommandAccess {
    use daemon_auth::Capability;
    match crate::request_context::current_principal() {
        Some(p) if p.has(Capability::AccessAdmin) || p.has(Capability::ControlWrite) => {
            CommandAccess::Admin
        }
        _ => CommandAccess::User,
    }
}

/// Whether `caller` may run a command requiring `required`. The `User` floor is always allowed;
/// `Admin` requires an admin caller.
pub fn access_allows(required: CommandAccess, caller: CommandAccess) -> bool {
    match required {
        CommandAccess::User => true,
        CommandAccess::Admin => caller == CommandAccess::Admin,
    }
}

/// A read-only, user-tier, session-scoped builtin spec starting point.
fn spec(name: &str, summary: &str, category: &str) -> CommandSpec {
    CommandSpec {
        name: name.to_string(),
        summary: summary.to_string(),
        category: category.to_string(),
        source: "node".to_string(),
        ..CommandSpec::default()
    }
}

/// The built-in command catalog: thin specs over existing typed `NodeApi` ops. No new op logic —
/// these only *advertise* (and route to) capabilities the node already has.
fn builtin_catalog() -> Vec<(CommandSpec, Builtin)> {
    let node = |mut s: CommandSpec| {
        s.scope = CommandScope::Node;
        s
    };
    let aliased = |mut s: CommandSpec, aliases: &[&str]| {
        s.aliases = aliases.iter().map(|a| a.to_string()).collect();
        s
    };
    let admin = |mut s: CommandSpec| {
        s.min_access = CommandAccess::Admin;
        s
    };
    let mutating = |mut s: CommandSpec| {
        s.side_effecting = true;
        s
    };
    vec![
        (
            aliased(
                node(spec("help", "List the available commands", "Info")),
                &["commands"],
            ),
            Builtin::Help,
        ),
        (
            node(spec(
                "whoami",
                "Show the node profile + partition identity",
                "Info",
            )),
            Builtin::Whoami,
        ),
        (
            node(spec("version", "Show the daemon build version", "Info")),
            Builtin::Version,
        ),
        (
            spec("status", "Show node + session status", "Info"),
            Builtin::Status,
        ),
        (
            node(aliased(
                spec("usage", "Show folded token usage + cost", "Info"),
                &["credits"],
            )),
            Builtin::Usage,
        ),
        (
            node(aliased(
                spec("sessions", "List the session roster", "Info"),
                &["agents", "tasks"],
            )),
            Builtin::Sessions,
        ),
        (
            mutating(aliased(
                spec("stop", "Cancel in-flight work for the session", "Session"),
                &["cancel"],
            )),
            Builtin::Stop,
        ),
        (
            mutating(spec("model", "Switch the session model", "Session")),
            Builtin::Model,
        ),
        (
            mutating(aliased(
                spec(
                    "mode",
                    "Set the session approval mode (yolo|fast|ask|deny)",
                    "Session",
                ),
                &["yolo", "fast"],
            )),
            Builtin::Mode,
        ),
        (
            mutating(spec("title", "Rename the session", "Session")),
            Builtin::Title,
        ),
        (
            admin(mutating(spec(
                "approve",
                "Approve a parked edit-approval request",
                "Session",
            ))),
            Builtin::Approve,
        ),
        (
            admin(mutating(spec(
                "deny",
                "Deny a parked edit-approval request",
                "Session",
            ))),
            Builtin::Deny,
        ),
    ]
}

/// Render the catalog as a help listing grouped by category (the `help`/`commands` output).
pub fn render_help(specs: &[CommandSpec]) -> String {
    let mut by_cat: HashMap<&str, Vec<&CommandSpec>> = HashMap::new();
    for s in specs {
        let cat = if s.category.is_empty() {
            "Other"
        } else {
            s.category.as_str()
        };
        by_cat.entry(cat).or_default().push(s);
    }
    let mut cats: Vec<&str> = by_cat.keys().copied().collect();
    cats.sort_unstable();
    let mut out = String::from("Available commands:\n");
    for cat in cats {
        out.push_str(&format!("\n{cat}:\n"));
        let mut cmds = by_cat.remove(cat).unwrap_or_default();
        cmds.sort_by(|a, b| a.name.cmp(&b.name));
        for c in cmds {
            let aliases = if c.aliases.is_empty() {
                String::new()
            } else {
                format!(" ({})", c.aliases.join(", "))
            };
            let hint = if c.args_hint.is_empty() {
                String::new()
            } else {
                format!(" {}", c.args_hint)
            };
            out.push_str(&format!(
                "  /{}{}{} — {}\n",
                c.name, aliases, hint, c.summary
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::{
        CommandCx, CommandError, CommandInvocation, CommandOutput, CommandProvider,
        CommandSpec as CoreSpec,
    };
    use std::sync::Arc;

    /// A provider standing in for a context/memory engine: it advertises one aliased command and a
    /// second whose name collides with a built-in, so we can assert folding + conflict handling.
    struct FakeProvider {
        name: &'static str,
        specs: Vec<CoreSpec>,
    }

    #[async_trait::async_trait]
    impl CommandProvider for FakeProvider {
        fn name(&self) -> &str {
            self.name
        }
        fn commands(&self) -> Vec<CoreSpec> {
            self.specs.clone()
        }
        async fn run_command(
            &self,
            _inv: &CommandInvocation,
            _cx: &CommandCx<'_>,
        ) -> std::result::Result<CommandOutput, CommandError> {
            Ok(CommandOutput::text("ok"))
        }
    }

    #[test]
    fn builtins_resolve_by_name_alias_and_case() {
        let reg = CommandRegistry::with_builtins();
        assert!(reg.resolve("help").is_some());
        assert!(reg.resolve("/help").is_some(), "leading slash ignored");
        assert!(reg.resolve("commands").is_some(), "alias of help");
        assert!(reg.resolve("STOP").is_some(), "case-insensitive");
        assert!(reg.resolve("cancel").is_some(), "alias of stop");
        assert!(reg.resolve("definitely-not-a-command").is_none());
    }

    #[test]
    fn provider_commands_fold_in_with_source_and_aliases() {
        let mut reg = CommandRegistry::with_builtins();
        reg.register_provider(Arc::new(FakeProvider {
            name: "lcm",
            specs: vec![CoreSpec::new("lcm")
                .summary("Lossless context")
                .category("Context")
                .alias("context")],
        }));
        let entry = reg.resolve("lcm").expect("lcm folded in");
        assert_eq!(
            entry.spec.source, "lcm",
            "source stamped with provider name"
        );
        assert!(matches!(entry.owner, Owner::Provider(_)));
        assert!(reg.resolve("context").is_some(), "provider alias resolves");
    }

    #[test]
    fn name_collision_with_builtin_is_rejected_first_wins() {
        let mut reg = CommandRegistry::with_builtins();
        // A provider claiming the built-in `stop` must not displace the built-in handler.
        reg.register_provider(Arc::new(FakeProvider {
            name: "rogue",
            specs: vec![CoreSpec::new("stop").summary("hijack")],
        }));
        let entry = reg.resolve("stop").expect("stop still present");
        assert!(
            matches!(entry.owner, Owner::Builtin(Builtin::Stop)),
            "built-in stop wins over the colliding provider command"
        );
    }

    #[test]
    fn access_gate_enforces_user_floor_and_admin_tier() {
        // Read-only `User` floor: anyone (user or admin) may run.
        assert!(access_allows(CommandAccess::User, CommandAccess::User));
        assert!(access_allows(CommandAccess::User, CommandAccess::Admin));
        // Admin-tier: only an admin caller.
        assert!(!access_allows(CommandAccess::Admin, CommandAccess::User));
        assert!(access_allows(CommandAccess::Admin, CommandAccess::Admin));
    }

    #[tokio::test]
    async fn caller_access_is_principal_driven_not_origin() {
        use crate::request_context::{with_request_context, RequestContext};
        use daemon_auth::{Principal, Role};

        // Fail-closed inversion: with no authenticated principal, the tier is the read-only `User`
        // floor (was `Admin` for `origin == None`).
        assert_eq!(caller_access(None), CommandAccess::User);

        // The local-trust system principal keeps the admin tier (the in-process / FFI / Unix path).
        let admin =
            with_request_context(RequestContext::system(), async { caller_access(None) }).await;
        assert_eq!(admin, CommandAccess::Admin);

        // A plain user principal stays on the floor...
        let viewer_user = with_request_context(
            RequestContext::authenticated(Principal::from_roles("u", "u", vec![Role::User]), None),
            async { caller_access(None) },
        )
        .await;
        assert_eq!(viewer_user, CommandAccess::User);

        // ...while an operator (holds `ControlWrite`) is elevated to the admin tier.
        let operator = with_request_context(
            RequestContext::authenticated(
                Principal::from_roles("o", "o", vec![Role::Operator]),
                None,
            ),
            async { caller_access(None) },
        )
        .await;
        assert_eq!(operator, CommandAccess::Admin);
    }

    /// `test_command_manager.c` `/command-manager/new`: a fresh registry is empty (zero commands),
    /// resolves nothing, and enumerates no specs.
    #[test]
    fn empty_registry_reports_empty() {
        let reg = CommandRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.specs().is_empty());
        assert!(reg.resolve("anything").is_none());
    }

    /// The built-in catalog is enumerable: `len` matches the advertised `specs()`, and the listing
    /// includes the canonical `help` command (the `command_list` payload a client renders).
    #[test]
    fn builtins_catalog_is_enumerable() {
        let reg = CommandRegistry::with_builtins();
        assert!(!reg.is_empty());
        let specs = reg.specs();
        assert_eq!(reg.len(), specs.len(), "len tracks the enumerated catalog");
        assert!(
            specs.iter().any(|s| s.name == "help"),
            "help is advertised in the catalog"
        );
    }

    /// `test_command_manager.c` `/command-manager/find-and-execute`: resolving a provider command
    /// yields its `Owner::Provider`, whose handler dispatches the invocation to the provider.
    #[tokio::test]
    async fn resolve_provider_command_executes_via_owner() {
        let mut reg = CommandRegistry::with_builtins();
        reg.register_provider(Arc::new(FakeProvider {
            name: "lcm",
            specs: vec![CoreSpec::new("lcm").summary("Lossless context")],
        }));
        let entry = reg.resolve("lcm").expect("lcm resolves");
        let Owner::Provider(provider) = &entry.owner else {
            panic!("expected a provider owner");
        };
        let inv = CommandInvocation {
            name: "lcm".into(),
            args: "arg1 arg2".into(),
            session: None,
        };
        let out = provider
            .run_command(&inv, &CommandCx::node())
            .await
            .expect("dispatch succeeds");
        assert_eq!(out.text, "ok", "the resolved owner ran the command");
    }

    /// Two providers both claiming `dup`: the first registrant wins and the catalog keeps a single
    /// `dup` entry (the daemon's first-wins discipline in place of libpurple's priority stacking /
    /// `find-all`).
    #[test]
    fn duplicate_provider_registration_first_wins() {
        let mut reg = CommandRegistry::new();
        reg.register_provider(Arc::new(FakeProvider {
            name: "p1",
            specs: vec![CoreSpec::new("dup").summary("first")],
        }));
        reg.register_provider(Arc::new(FakeProvider {
            name: "p2",
            specs: vec![CoreSpec::new("dup").summary("second")],
        }));
        let entry = reg.resolve("dup").expect("dup resolves");
        assert_eq!(entry.spec.source, "p1", "first registrant wins");
        assert_eq!(
            reg.specs().iter().filter(|s| s.name == "dup").count(),
            1,
            "exactly one dup entry (no stacking)"
        );
    }

    /// A provider whose *alias* (not name) collides with an existing entry is rejected whole: the
    /// command does not partially register, and the incumbent alias still resolves to it.
    #[test]
    fn provider_alias_collision_rejects_whole_entry() {
        let mut reg = CommandRegistry::with_builtins();
        // `commands` is the built-in alias of `help`; a provider aliasing it must be rejected whole.
        reg.register_provider(Arc::new(FakeProvider {
            name: "rogue",
            specs: vec![CoreSpec::new("newcmd").summary("hijack").alias("commands")],
        }));
        assert!(
            reg.resolve("newcmd").is_none(),
            "the whole entry is rejected on an alias collision, not partially added"
        );
        let commands = reg.resolve("commands").expect("incumbent alias intact");
        assert!(matches!(commands.owner, Owner::Builtin(Builtin::Help)));
    }
}
